from app.datasets.pdf_loader import extract_pdf_text
from pathlib import Path

pdf_path = Path(
    "data/raw/resume/pdf+gold/Resume-Academic01.pdf"
)

text = extract_pdf_text(str(pdf_path))

print(text[:3000])