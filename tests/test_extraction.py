from pathlib import Path

from app.datasets.pdf_loader import extract_pdf_text

from app.datasets.schemas import (
    DatasetSample,
    FieldSpec
)

from app.extraction.extractor import Extractor
from app.llm.ollama_client import OllamaClient

from app.llm.prompts import (
    build_seed_prompt
)


# ------------------------------------------------
# Load PDF
# ------------------------------------------------

pdf_path = Path(
    "data/raw/resume/pdf+gold/Resume-Academic01.pdf"
)

document_text = extract_pdf_text(str(pdf_path))[:2500]


# ------------------------------------------------
# Create DatasetSample
# ------------------------------------------------

sample = DatasetSample(
    id="resume_001",
    input_text=document_text,
    ground_truth={}
)


# ------------------------------------------------
# Define extraction fields
# ------------------------------------------------

fields = [

    FieldSpec(
        name="full_name",
        type="string_exact",
        required=False
    ),

    FieldSpec(
        name="education",
        type="array_llm",
        required=False
    ),

    FieldSpec(
        name="research_areas",
        type="array_llm",
        required=False
    ),
]


# ------------------------------------------------
# Build prompt
# ------------------------------------------------

prompt_template = build_seed_prompt(
    [f.model_dump() for f in fields]
)


# ------------------------------------------------
# Setup model config
# ------------------------------------------------

model_config = {
    "ollama_model": "mistral",
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 2048
}


# ------------------------------------------------
# Create extractor
# ------------------------------------------------

client = OllamaClient()

extractor = Extractor(
    client=client,
    model_config=model_config
)


# ------------------------------------------------
# Run extraction
# ------------------------------------------------

result = extractor.extract_sample(
    prompt_template=prompt_template,
    sample=sample,
    fields=fields
)


# ------------------------------------------------
# Print results
# ------------------------------------------------

print("\nPARSE SUCCESS:\n")
print(result.parse_success)

print("\nRAW OUTPUT:\n")
print(result.raw_output)

print("\nPARSED OUTPUT:\n")
print(result.parsed)