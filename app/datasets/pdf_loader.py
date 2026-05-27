import fitz
import pdfplumber
import pytesseract

from pdf2image import convert_from_path


# SET YOUR TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_pdf_text(pdf_path: str) -> str:

    text = ""

    # -------------------------------------------------
    # TRY PYMUPDF
    # -------------------------------------------------
    try:

        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

    except Exception as e:
        print("PyMuPDF failed:", e)

    # -------------------------------------------------
    # TRY PDFPLUMBER
    # -------------------------------------------------
    if not text.strip():

        print("Using pdfplumber fallback...")

        try:

            with pdfplumber.open(pdf_path) as pdf:

                for page in pdf.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

        except Exception as e:
            print("pdfplumber failed:", e)

    # -------------------------------------------------
    # OCR FALLBACK
    # -------------------------------------------------
    if not text.strip():

        print("Using OCR fallback...")

        try:

            images = convert_from_path(
            pdf_path,
            poppler_path=r"C:\poppler\poppler-26.02.0\Library\bin"
            )

            for image in images:

                ocr_text = pytesseract.image_to_string(image)

                text += ocr_text + "\n"

        except Exception as e:
            print("OCR failed:", e)

    return text