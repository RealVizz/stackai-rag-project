import pdfplumber
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> str:
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text
    except Exception:
        return ""  # Todo : think about proper Exception and api error response/err code to return in such case.


def simple_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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

    if not full_text.strip():
        return []

    return simple_chunker(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

