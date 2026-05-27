"""
Prompt templates for extraction and optimization tasks.
Seed prompts are the starting point for the optimizer.
"""

from __future__ import annotations

from string import Template
from typing import Any
from app.datasets.schemas import FieldSpec

# ---------------------------------------------------------------------------
# Extraction seed prompts
# ---------------------------------------------------------------------------

SEED_EXTRACTION_PROMPT = """\
You are a precise data extraction assistant.

Given the following document, extract the requested fields and return them as a JSON object.
Only include fields that are explicitly present in the document.
If a field cannot be found, set its value to null.

Fields to extract:
{field_definitions}

Document:
{document}

Return ONLY a valid JSON object. Do not include any explanation or markdown formatting.
"""

EXTRACTION_PROMPT_WITH_SCHEMA = """\
You are a strict JSON information extraction system.

Your ONLY task is to extract the requested fields from the document.

CRITICAL RULES:
- Return ONLY valid JSON
- No markdown
- No explanations
- No comments
- No placeholder text
- No extra fields
- No trailing commas
- Do NOT invent fields
- Do NOT hallucinate values
- If information is missing, return null
- Output MUST be parseable by Python json.loads()

Extract ONLY the fields defined in this schema.

Schema:
{schema}

Document:
{document}

Return ONLY the JSON object.
"""

EXTRACTION_PROMPT_COT = """\
You are a careful data extraction assistant.

Let's think step by step before extracting.

First, identify all relevant mentions of each field in the document.
Then, extract only explicitly stated values.
Finally, format the output as a JSON object.

Fields required:
{field_definitions}

Document:
{document}

Step-by-step reasoning:
[Analyze each field briefly]

JSON output:
"""

FEW_SHOT_HEADER = """\
You are a precise data extraction assistant.

Here are examples of correct extractions:

{examples}

Now extract from the following document:
Fields: {field_definitions}

Document:
{document}

Return ONLY a valid JSON object.
"""

# ---------------------------------------------------------------------------
# Optimizer / mutator prompts
# ---------------------------------------------------------------------------

MUTATION_SYSTEM_PROMPT = """\
You are an expert prompt engineer specializing in structured data extraction tasks.
Your job is to improve extraction prompts to maximize accuracy.
You understand JSON schemas, LLM behavior, and extraction failure modes.
"""

INSTRUCTION_REWRITE_PROMPT = """\
You are a prompt optimization expert. Rewrite the following extraction prompt to be clearer, more precise, and better at preventing hallucination.

Original prompt:
{original_prompt}

Context:
- Task: Extract structured fields from documents
- Fields: {field_names}
- Known failures: {failure_examples}

Rules for rewriting:
1. Keep all field names exactly as specified
2. Make instructions unambiguous
3. Reinforce that null should be returned for missing fields
4. Do not add new fields or remove existing ones
5. Keep the output format as JSON

Return ONLY the improved prompt text, no explanation.
"""

OUTPUT_FORMAT_TIGHTEN_PROMPT = """\
Improve the output formatting instructions in this extraction prompt to ensure the LLM returns valid, parseable JSON every time.

Original prompt:
{original_prompt}

Improve:
- JSON formatting requirements
- Field type constraints
- Error prevention for common JSON mistakes

Return ONLY the improved prompt text.
"""

VERBOSITY_REDUCE_PROMPT = """\
Make this extraction prompt more concise while preserving all critical instructions.
Target: reduce word count by approximately 30% without losing precision.

Original prompt:
{original_prompt}

Return ONLY the condensed prompt text.
"""

HALLUCINATION_SUPPRESS_PROMPT = """\
Strengthen this extraction prompt to prevent the LLM from hallucinating or guessing field values.

Original prompt:
{original_prompt}

Add clear instructions that:
1. Prohibit inferring missing values
2. Require null for absent fields
3. Prevent the model from filling gaps with plausible-sounding values

Return ONLY the improved prompt text.
"""

SCHEMA_AWARE_REFINE_PROMPT = """\
Refine this extraction prompt to be more aware of the field types and formats required.

Original prompt:
{original_prompt}

Field schema:
{field_schema}

Add:
- Type hints for each field (string, number, array, etc.)
- Format examples where helpful
- Validation reminders

Return ONLY the improved prompt text.
"""

FIELD_CONSTRAINT_PROMPT = """\
Add field-specific constraints to this extraction prompt to improve per-field accuracy.

Original prompt:
{original_prompt}

Fields with low accuracy:
{weak_fields}

For each weak field, add a specific instruction about what to look for and what to avoid.

Return ONLY the improved prompt text.
"""

COT_TOGGLE_ADD_PROMPT = """\
Add a chain-of-thought reasoning step to this extraction prompt to improve accuracy on complex fields.

Original prompt:
{original_prompt}

Insert a step-by-step reasoning instruction before the final JSON output.

Return ONLY the improved prompt text.
"""

COT_TOGGLE_REMOVE_PROMPT = """\
Remove any chain-of-thought or step-by-step reasoning instructions from this prompt.
Make it direct and concise, going straight to JSON output.

Original prompt:
{original_prompt}

Return ONLY the modified prompt text.
"""




# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_field_definitions(fields: list[Any]) -> str:
    """Format a list of field configs into a human-readable list."""
    
    lines = []

    for field in fields:

        if isinstance(field, dict):
            name = field.get("name")
            ftype = field.get("type", "string_exact")
            required = field.get("required", True)

        else:
            name = field.name
            ftype = field.type
            required = field.required

        req_text = "required" if required else "optional"

        lines.append(f"- {name} ({ftype}, {req_text})")

    return "\n".join(lines)


def format_schema_json(fields: list[Any]) -> str:
    """Format fields into a JSON schema-like string."""

    import json

    schema: dict[str, Any] = {}

    type_map = {
        "string_exact": "string",
        "string_semantic": "string",
        "integer_exact": "integer",
        "number_tolerance": "number",
        "array_llm": "array",
    }

    for field in fields:

        if isinstance(field, dict):
            name = field.get("name")
            field_type = field.get("type", "string_exact")
            required = field.get("required", True)

        else:
            name = field.name
            field_type = field.type
            required = field.required

        ftype = type_map.get(field_type, "string")

        schema[name] = {
            "type": ftype,
            "required": required,
        }

    return json.dumps(schema, indent=2)


def build_seed_prompt(fields: list[Any], use_schema: bool = True) -> str:
    """
    Build the initial seed prompt template.
    """

    if use_schema:
        return EXTRACTION_PROMPT_WITH_SCHEMA.replace(
            "{schema}",
            format_schema_json(fields)
        )

    return SEED_EXTRACTION_PROMPT.replace(
        "{field_definitions}",
        format_field_definitions(fields)
    )