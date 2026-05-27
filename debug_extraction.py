from app.datasets.pdf_loader import extract_pdf_text
from app.llm.ollama_client import OllamaClient

pdf_path = "data/raw/resume/pdf+gold/Resume-Med.pdf"

text = extract_pdf_text(pdf_path)

print("\nTEXT EXTRACTED:\n")
print(text[:1000])

prompt = f"""
You are a strict JSON information extraction system.

Extract ONLY the following fields from the resume.

Return STRICT VALID JSON ONLY.

RULES:
- No markdown
- No explanations
- No comments
- No extra fields
- No trailing commas
- If missing, use null

Required JSON structure:

{{
  "full_name": "string",
  "education": [],
  "research_areas": []
}}

Resume Text:
{text[:6000]}

Return ONLY valid JSON.
"""

client = OllamaClient()

response = client.generate(
    model="llama3",
    prompt=prompt
)

print("\nMODEL OUTPUT:\n")
print(response["response"])