"""
LLM-based structured field extractor.
Applies a prompt to a document and returns raw + parsed extraction results.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.datasets.schemas import DatasetSample, ExtractionResult, FieldSpec
from app.llm.ollama_client import OllamaClient
from app.llm.roles import get_role
from app.persistence.cache import ResponseCache
from app.utils.hashing import hash_prompt_input, short_hash
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Extractor:
    """
    Applies an extraction prompt to documents using an Ollama model.

    Caches LLM responses to avoid redundant calls during evaluation.
    """

    def __init__(
        self,
        client: OllamaClient,
        model_config: dict[str, Any],
        cache: Optional[ResponseCache] = None,
    ) -> None:
        self._client = client
        self._model_cfg = model_config
        self._cache = cache
        self._model_name = model_config["ollama_model"]

    def extract_sample(
        self,
        prompt_template: str,
        sample: DatasetSample,
        fields: list[FieldSpec],
    ) -> ExtractionResult:
        """
        Apply the extraction prompt to a single document.

        Args:
            prompt_template: The extraction prompt (may contain {document} placeholder).
            sample: The dataset sample to extract from.
            fields: Field specs (used for schema context).

        Returns:
            ExtractionResult with raw output and parsed dict.
        """
        prompt = self._build_prompt(prompt_template, sample.input_text)
        prompt_hash = short_hash(prompt)
        cache_key = hash_prompt_input(prompt, sample.input_text, self._model_name)

        # Check cache first
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for sample {sample.id}")
                
                print("\nCACHED RAW MODEL OUTPUT:\n")
                print(cached["raw_output"])

                print("\nCACHED PARSED OUTPUT:\n")
                print(cached.get("parsed"))

                return ExtractionResult(
                    sample_id=sample.id,
                    raw_output=cached["raw_output"],
                    parsed=cached.get("parsed"),
                    parse_success=cached.get("parse_success", False),
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    latency_ms=0.0,  # cached, no actual latency
                    model=self._model_name,
                    prompt_hash=prompt_hash,
                )

        # Call LLM
        try:
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                system=get_role("extractor"),
                temperature=self._model_cfg.get("temperature", 0.0),
                top_p=self._model_cfg.get("top_p", 0.9),
                max_tokens=self._model_cfg.get("max_tokens", 2048),
            )
        except Exception as exc:
            logger.error(f"LLM call failed for sample {sample.id}: {exc}")
            return ExtractionResult(
                sample_id=sample.id,
                raw_output="",
                parsed=None,
                parse_success=False,
                model=self._model_name,
                prompt_hash=prompt_hash,
            )

        raw_output = response["response"]
        parsed, parse_success = self._parse_json(raw_output)
        print("\nRAW MODEL OUTPUT:\n")
        print(raw_output)

        print("\nPARSED OUTPUT:\n")
        print(parsed)

        result = ExtractionResult(
            sample_id=sample.id,
            raw_output=raw_output,
            parsed=parsed,
            parse_success=parse_success,
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            latency_ms=response["latency_ms"],
            model=self._model_name,
            prompt_hash=prompt_hash,
        )

        # Store in cache
        if self._cache:
            self._cache.set(
                cache_key,
                {
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "parse_success": parse_success,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                },
            )

        return result

    def extract_batch(
        self,
        prompt_template: str,
        samples: list[DatasetSample],
        fields: list[FieldSpec],
    ) -> list[ExtractionResult]:
        """
        Extract fields from a batch of samples.

        Args:
            prompt_template: Extraction prompt template.
            samples: List of samples to process.
            fields: Field specifications.

        Returns:
            List of ExtractionResult in the same order as samples.
        """
        results = []
        for i, sample in enumerate(samples):
            logger.debug(f"Extracting sample {i+1}/{len(samples)}: {sample.id}")
            result = self.extract_sample(prompt_template, sample, fields)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(template: str, document: str) -> str:
        """Insert document text into the prompt template."""
        if "{document}" in template:
            return template.replace("{document}", document)
        # Append document if no placeholder
        return f"{template}\n\nDocument:\n{document}"

    @staticmethod
    def _parse_json(raw: str) -> tuple[Optional[dict[str, Any]], bool]:
        """
        Attempt to parse JSON from LLM output.
        Handles common issues: markdown code fences, leading/trailing text.

        Returns:
            Tuple of (parsed_dict or None, success_bool).
        """
        if not raw or not raw.strip():
            return None, False

        # Strip markdown code fences
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        cleaned = cleaned.rstrip("`").strip()

# Remove JS-style comments
        cleaned = re.sub(r"//.*", "", cleaned)

# Remove trailing commas before ] or }
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

        # Try direct parse
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj, True
        except json.JSONDecodeError:
            pass

        # Try extracting first JSON object from mixed text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    return obj, True
            except json.JSONDecodeError:
                pass

        logger.debug(f"JSON parse failed. Raw output snippet: {raw[:200]!r}")
        return None, False
