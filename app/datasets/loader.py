"""
Dataset loader supporting JSON files, ExtractBench format,
PDF+gold datasets, and synthetic generation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from app.datasets.schemas import DatasetSample, FieldSpec
from app.datasets.pdf_loader import extract_pdf_text
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """
    Loads dataset samples from various sources.

    Supported formats:
    - JSON
    - PDF + .gold.json
    - Synthetic generation
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        random.seed(seed)

    def load(
        self,
        dataset_config: dict[str, Any]
    ) -> list[DatasetSample]:

        ds_type = dataset_config.get(
            "type",
            "extractbench"
        )

        path = dataset_config.get("path")

        if ds_type == "synthetic":

            logger.info(
                f"Generating synthetic dataset: "
                f"{dataset_config.get('name')}"
            )

            return self._generate_synthetic(
                dataset_config
            )

        if not path:
            raise ValueError(
                f"Dataset config missing 'path' "
                f"for type '{ds_type}'"
            )

        fpath = Path(path)

        if not fpath.exists():

            logger.warning(
                f"Dataset file not found at '{fpath}'. "
                f"Falling back to synthetic generation."
            )

            return self._generate_synthetic(
                dataset_config
            )

        logger.info(
            f"Loading dataset from '{fpath}'"
        )

        dataset_format = dataset_config.get(
            "format",
            "json"
        )

        if dataset_format == "pdf_gold":

            return self._load_pdf_gold(
                fpath,
                dataset_config
            )

        return self._load_json(
            fpath,
            dataset_config
        )

    def get_field_specs(
        self,
        dataset_config: dict[str, Any]
    ) -> list[FieldSpec]:

        raw_fields = dataset_config.get(
            "fields_to_extract",
            []
        )

        return [FieldSpec(**f) for f in raw_fields]

    # ---------------------------------------------------------
    # JSON LOADER
    # ---------------------------------------------------------

    def _load_json(
        self,
        path: Path,
        dataset_config: dict[str, Any]
    ) -> list[DatasetSample]:

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):

            for key in (
                "samples",
                "data",
                "records",
                "examples",
            ):
                if key in data:
                    data = data[key]
                    break

        input_field = dataset_config.get(
            "input_field",
            "text"
        )

        samples: list[DatasetSample] = []

        for i, record in enumerate(data):

            sample_id = str(
                record.get(
                    "id",
                    f"sample_{i:04d}"
                )
            )

            input_text = record.get(
                input_field,
                ""
            )

            ground_truth = record.get(
                "ground_truth",
                {}
            )

            if not ground_truth:

                exclude = {
                    input_field,
                    "id",
                    "source",
                }

                ground_truth = {
                    k: v
                    for k, v in record.items()
                    if k not in exclude
                }

            samples.append(
                DatasetSample(
                    id=sample_id,
                    input_text=str(input_text),
                    ground_truth=ground_truth,
                    source=str(path),
                )
            )

        logger.info(
            f"Loaded {len(samples)} samples "
            f"from '{path}'"
        )

        return samples

    # ---------------------------------------------------------
    # PDF + GOLD LOADER
    # ---------------------------------------------------------

    def _load_pdf_gold(
        self,
        path: Path,
        dataset_config: dict[str, Any]
    ) -> list[DatasetSample]:

        samples: list[DatasetSample] = []

        pdf_files = list(path.glob("*.pdf"))

        logger.info(
            f"Found {len(pdf_files)} PDF files"
        )

        for pdf_file in pdf_files:

            gold_file = pdf_file.with_suffix(
                ".gold.json"
            )

            if not gold_file.exists():

                logger.warning(
                    f"Missing gold file for "
                    f"{pdf_file.name}"
                )

                continue

            try:

                text = extract_pdf_text(
                    str(pdf_file)
                )

                with open(
                    gold_file,
                    encoding="utf-8"
                ) as f:

                    gold_json = json.load(f)

                samples.append(
                    DatasetSample(
                        id=pdf_file.stem,
                        input_text=text,
                        ground_truth=gold_json,
                        source=str(pdf_file),
                    )
                )

            except Exception as e:

                logger.error(
                    f"Failed loading "
                    f"{pdf_file.name}: {e}"
                )

        logger.info(
            f"Loaded {len(samples)} "
            f"PDF+gold samples"
        )

        return samples

    # ---------------------------------------------------------
    # SYNTHETIC GENERATION
    # ---------------------------------------------------------

    def _generate_synthetic(
        self,
        dataset_config: dict[str, Any]
    ) -> list[DatasetSample]:

        n = dataset_config.get(
            "num_samples",
            50
        )

        fields = dataset_config.get(
            "fields_to_extract",
            []
        )

        samples: list[DatasetSample] = []

        for i in range(n):

            ground_truth, input_text = (
                self._generate_synthetic_sample(
                    fields,
                    i
                )
            )

            samples.append(
                DatasetSample(
                    id=f"synth_{i:04d}",
                    input_text=input_text,
                    ground_truth=ground_truth,
                    source="synthetic",
                    metadata={"generated": True},
                )
            )

        logger.info(
            f"Generated {len(samples)} "
            f"synthetic samples"
        )

        return samples

    def _generate_synthetic_sample(
        self,
        fields: list[dict[str, Any]],
        idx: int
    ) -> tuple[dict[str, Any], str]:

        ground_truth: dict[str, Any] = {}
        snippets: list[str] = []

        vendors = [
            "Acme Corp",
            "Global Solutions Ltd",
            "TechWave Inc",
            "Sunrise Supplies",
        ]

        tags_pool = [
            "electronics",
            "software",
            "hardware",
            "consulting",
            "support",
            "cloud",
        ]

        for field in fields:

            name = field["name"]

            ftype = field.get(
                "type",
                "string_exact"
            )

            if ftype == "string_exact":

                value = (
                    f"{name.upper()}-"
                    f"{idx:04d}-"
                    f"{random.randint(1000,9999)}"
                )

                ground_truth[name] = value

                snippets.append(
                    f"The {name.replace('_', ' ')} "
                    f"is {value}."
                )

            elif ftype == "string_semantic":

                value = random.choice(vendors)

                ground_truth[name] = value

                snippets.append(
                    f"{name.replace('_', ' ').title()}: "
                    f"{value}"
                )

            elif ftype == "integer_exact":

                value = random.randint(1, 20)

                ground_truth[name] = value

                snippets.append(
                    f"{name.replace('_', ' ').title()}: "
                    f"{value} years"
                )

            elif ftype == "number_tolerance":

                value = round(
                    random.uniform(10.0, 9999.99),
                    2
                )

                ground_truth[name] = value

                snippets.append(
                    f"Total {name.replace('_', ' ')}: "
                    f"${value:.2f}"
                )

            elif ftype == "array_llm":

                count = random.randint(2, 5)

                value = random.sample(
                    tags_pool,
                    min(count, len(tags_pool))
                )

                ground_truth[name] = value

                snippets.append(
                    f"{name.replace('_', ' ').title()}: "
                    f"{', '.join(value)}."
                )

        filler = [
            "This document was processed on behalf of the client.",
            "Please review all information carefully.",
            "Contact support if you have any questions.",
        ]

        random.shuffle(snippets)
        random.shuffle(filler)

        document = " ".join(
            snippets[:3] +
            filler[:2] +
            snippets[3:]
        )

        return ground_truth, document