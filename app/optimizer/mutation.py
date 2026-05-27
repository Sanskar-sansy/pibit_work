"""
Prompt mutation strategies for the optimization loop.

Each strategy takes a prompt text and context, calls an LLM mutator,
and returns a new prompt variant.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from app.datasets.schemas import FieldSpec
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import (
    COT_TOGGLE_ADD_PROMPT,
    COT_TOGGLE_REMOVE_PROMPT,
    FIELD_CONSTRAINT_PROMPT,
    HALLUCINATION_SUPPRESS_PROMPT,
    INSTRUCTION_REWRITE_PROMPT,
    OUTPUT_FORMAT_TIGHTEN_PROMPT,
    SCHEMA_AWARE_REFINE_PROMPT,
    VERBOSITY_REDUCE_PROMPT,
    format_field_definitions,
    format_schema_json,
)
from app.llm.roles import get_role
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, "MutationStrategy"] = {}


def register_strategy(name: str):
    """Decorator to register a mutation strategy by name."""
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls()
        return cls
    return decorator


class MutationStrategy:
    """Abstract base for a prompt mutation strategy."""

    name: str = "base"

    def mutate(
        self,
        prompt: str,
        client: OllamaClient,
        model_config: dict[str, Any],
        fields: list[FieldSpec],
        context: dict[str, Any],
    ) -> Optional[str]:
        """
        Generate a mutated version of the prompt.

        Args:
            prompt: The current prompt text.
            client: Ollama client for LLM calls.
            model_config: Mutator model config.
            fields: Field specs for context.
            context: Additional context (failure examples, weak fields, etc.).

        Returns:
            Mutated prompt string, or None if mutation failed.
        """
        raise NotImplementedError

    def _call_mutator(
        self,
        mutation_prompt: str,
        client: OllamaClient,
        model_config: dict[str, Any],
    ) -> Optional[str]:
        """Call the LLM mutator and return its response."""
        try:
            response = client.generate(
                model=model_config["ollama_model"],
                prompt=mutation_prompt,
                system=get_role("mutator"),
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 0.95),
                max_tokens=model_config.get("max_tokens", 1024),
            )
            result = response["response"].strip()
            if not result:
                logger.warning(f"[{self.name}] Mutator returned empty response")
                return None
            return result
        except Exception as exc:
            logger.error(f"[{self.name}] Mutation call failed: {exc}")
            return None


@register_strategy("instruction_rewrite")
class InstructionRewriteStrategy(MutationStrategy):
    """Rewrite the overall instructions for clarity and precision."""

    name = "instruction_rewrite"

    def mutate(self, prompt, client, model_config, fields, context):
        field_names = [f.name for f in fields]
        failures = context.get("failure_examples", "")
        mutation_prompt = INSTRUCTION_REWRITE_PROMPT.format(
            original_prompt=prompt,
            field_names=", ".join(field_names),
            failure_examples=failures or "No specific failures recorded.",
        )
        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("output_format_tighten")
class OutputFormatTightenStrategy(MutationStrategy):
    """Tighten the JSON output format instructions."""

    name = "output_format_tighten"

    def mutate(self, prompt, client, model_config, fields, context):
        mutation_prompt = OUTPUT_FORMAT_TIGHTEN_PROMPT.format(
            original_prompt=prompt
        )
        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("verbosity_reduce")
class VerbosityReduceStrategy(MutationStrategy):
    """Reduce prompt verbosity by ~30% while keeping critical instructions."""

    name = "verbosity_reduce"

    def mutate(self, prompt, client, model_config, fields, context):
        mutation_prompt = VERBOSITY_REDUCE_PROMPT.format(original_prompt=prompt)
        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("hallucination_suppress")
class HallucinationSuppressStrategy(MutationStrategy):
    """Add explicit instructions to suppress hallucinated values."""

    name = "hallucination_suppress"

    def mutate(self, prompt, client, model_config, fields, context):
        # Also try direct rule injection without LLM for reliability
        suppression_rules = (
            "\n\nCRITICAL RULES:\n"
            "- Extract ONLY information explicitly stated in the document.\n"
            "- Do NOT infer, guess, or generate plausible-sounding values.\n"
            "- If a field is absent from the document, return null for that field.\n"
            "- Do NOT fill missing fields with examples or assumptions.\n"
        )

        # Try LLM-based suppression first
        mutation_prompt = HALLUCINATION_SUPPRESS_PROMPT.format(original_prompt=prompt)
        result = self._call_mutator(mutation_prompt, client, model_config)

        if result:
            return result

        # Fallback: direct injection
        return prompt + suppression_rules


@register_strategy("schema_aware_refine")
class SchemaAwareRefineStrategy(MutationStrategy):
    """Refine prompt with field type hints and format examples."""

    name = "schema_aware_refine"

    def mutate(self, prompt, client, model_config, fields, context):
        schema = format_schema_json(fields)
        mutation_prompt = SCHEMA_AWARE_REFINE_PROMPT.format(
            original_prompt=prompt, field_schema=schema
        )
        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("field_constraint_add")
class FieldConstraintStrategy(MutationStrategy):
    """Add targeted constraints for the weakest fields."""

    name = "field_constraint_add"

    def mutate(self, prompt, client, model_config, fields, context):
        per_field_scores = context.get("per_field_scores", {})
        if not per_field_scores:
            # No score data -> pick 2 random fields
            weak = [f.name for f in random.sample(fields, min(2, len(fields)))]
        else:
            # Pick bottom 2 fields
            sorted_fields = sorted(per_field_scores.items(), key=lambda x: x[1])
            weak = [name for name, _ in sorted_fields[:2]]

        weak_field_desc = "\n".join(
            f"- {name}: score={per_field_scores.get(name, 'unknown'):.3f}"
            if isinstance(per_field_scores.get(name), float)
            else f"- {name}"
            for name in weak
        )

        mutation_prompt = FIELD_CONSTRAINT_PROMPT.format(
            original_prompt=prompt, weak_fields=weak_field_desc
        )
        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("chain_of_thought_toggle")
class ChainOfThoughtToggleStrategy(MutationStrategy):
    """Toggle chain-of-thought reasoning in the prompt."""

    name = "chain_of_thought_toggle"

    _COT_MARKERS = ["step by step", "let's think", "reasoning:", "chain of thought"]

    def mutate(self, prompt, client, model_config, fields, context):
        has_cot = any(m in prompt.lower() for m in self._COT_MARKERS)

        if has_cot:
            # Remove CoT
            mutation_prompt = COT_TOGGLE_REMOVE_PROMPT.format(original_prompt=prompt)
        else:
            # Add CoT
            mutation_prompt = COT_TOGGLE_ADD_PROMPT.format(original_prompt=prompt)

        return self._call_mutator(mutation_prompt, client, model_config)


@register_strategy("few_shot_insert")
class FewShotInsertStrategy(MutationStrategy):
    """
    Insert few-shot examples from the training set into the prompt.
    Falls back to synthetic examples if no training samples are in context.
    """

    name = "few_shot_insert"

    def mutate(self, prompt, client, model_config, fields, context):
        train_samples = context.get("train_samples", [])
        num_examples = context.get("num_few_shot", 2)

        if train_samples:
            # Use real examples from training set
            selected = random.sample(train_samples, min(num_examples, len(train_samples)))
            examples_text = self._format_examples(selected, fields)
        else:
            # Inject a structural placeholder
            examples_text = self._synthetic_example(fields)

        if "Example" in prompt or "example" in prompt:
            # Prompt already has examples; don't double-insert
            return None

        # Insert examples after the first paragraph
        lines = prompt.split("\n")
        insert_idx = min(5, len(lines))
        example_block = f"\n\nExamples:\n{examples_text}\n"
        new_lines = lines[:insert_idx] + [example_block] + lines[insert_idx:]
        return "\n".join(new_lines)

    @staticmethod
    def _format_examples(samples: list, fields: list[FieldSpec]) -> str:
        lines = []
        for i, s in enumerate(samples, 1):
            import json
            gt = {f.name: s.ground_truth.get(f.name) for f in fields}
            lines.append(
                f"Example {i}:\nDocument: {s.input_text[:200]}...\n"
                f"Output: {json.dumps(gt, ensure_ascii=False)}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _synthetic_example(fields: list[FieldSpec]) -> str:
        import json
        placeholders = {}
        for f in fields:
            if f.type in ("string_exact", "string_semantic"):
                placeholders[f.name] = "Example Value"
            elif f.type == "integer_exact":
                placeholders[f.name] = 5
            elif f.type == "number_tolerance":
                placeholders[f.name] = 123.45
            elif f.type == "array_llm":
                placeholders[f.name] = ["item1", "item2"]
            else:
                placeholders[f.name] = "value"
        return (
            f"Example 1:\nDocument: [sample document text here]\n"
            f"Output: {json.dumps(placeholders, ensure_ascii=False)}"
        )


def get_strategy(name: str) -> MutationStrategy:
    """Retrieve a registered mutation strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown mutation strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())
