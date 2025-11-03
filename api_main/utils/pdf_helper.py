import pdfplumber

from config.config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFProcessingError(Exception):
    pass


def extract_text_from_pdf(file_path: str) -> str:
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text.strip() + "\n"

        if not full_text.strip():
            raise PDFProcessingError("No text could be extracted from the PDF.")

        return full_text

    except Exception as e:
        print(f"Failed to process PDF file {file_path}: {e}")
        raise PDFProcessingError(f"Failed to extract text from PDF: {e}")


def simple_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not text.strip():
        return []

    chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += (chunk_size - chunk_overlap)

    return chunks


def process_pdf(file_path: str) -> list[str]:
    full_text = extract_text_from_pdf(file_path)
    return simple_chunker(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

