from unittest.mock import patch, MagicMock

import pytest

from api_main.utils.pdf_helper import (
    simple_chunker,
    extract_text_from_pdf,
    process_pdf,
    PDFProcessingError
)
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def test_simple_chunker_with_overlap():
    """Tests chunking a standard string with overlap."""
    text_to_chunk = "This is a test text for the simple chunker function."
    chunks = simple_chunker(text_to_chunk, chunk_size=20, chunk_overlap=5)
    expected_chunks = [
        "This is a test text ",
        "text for the simple ",
        "mple chunker functio",
        "nction."
    ]
    assert chunks == expected_chunks


def test_simple_chunker_no_overlap():
    """Tests chunking with zero overlap."""
    text_to_chunk = "abcde fghij klmno"
    chunks = simple_chunker(text_to_chunk, chunk_size=5, chunk_overlap=0)
    expected_chunks = ["abcde", " fghi", "j klm", "no"]
    assert chunks == expected_chunks


def test_simple_chunker_text_smaller_than_chunk_size():
    """Tests when the text is shorter than the specified chunk size."""
    text_to_chunk = "short text"
    chunks = simple_chunker(text_to_chunk, chunk_size=20, chunk_overlap=5)
    assert chunks == ["short text"]


def test_simple_chunker_empty_text():
    """Tests if an empty string results in an empty list of chunks."""
    text_to_chunk = ""
    chunks = simple_chunker(text_to_chunk, chunk_size=100, chunk_overlap=10)
    assert chunks == []


def test_simple_chunker_exact_multiple_with_overlap():
    """Tests chunking when text length is an exact multiple of the step size."""
    text_to_chunk = "123456789012345"  # 15 chars
    chunks = simple_chunker(text_to_chunk, chunk_size=5, chunk_overlap=2)
    assert chunks == ["12345", "45678", "78901", "01234", "345"]


def test_simple_chunker_invalid_overlap_raises_error():
    """Tests that an invalid overlap (>= chunk_size) raises a ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size."):
        simple_chunker("some text", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size."):
        simple_chunker("some text", chunk_size=10, chunk_overlap=11)


@patch('api_main.utils.pdf_helper.pdfplumber.open')
def test_extract_text_from_pdf_success(mock_pdfplumber_open):
    """Tests successful text extraction from a mocked PDF with multiple pages."""
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Hello world from page 1. "
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "This is page 2."

    mock_pdf_object = MagicMock()
    mock_pdf_object.pages = [mock_page1, mock_page2]

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_pdf_object
    mock_context_manager.__exit__.return_value = None
    mock_pdfplumber_open.return_value = mock_context_manager

    extracted_text = extract_text_from_pdf("dummy/path/to/file.pdf")

    assert extracted_text == "Hello world from page 1.\nThis is page 2.\n"
    mock_pdfplumber_open.assert_called_once_with("dummy/path/to/file.pdf")


@patch('api_main.utils.pdf_helper.pdfplumber.open')
def test_extract_text_from_pdf_no_text_extracted(mock_pdfplumber_open):
    """Tests that a PDFProcessingError is raised for PDFs containing no text."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_pdf_object = MagicMock()
    mock_pdf_object.pages = [mock_page]

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_pdf_object
    mock_context_manager.__exit__.return_value = None
    mock_pdfplumber_open.return_value = mock_context_manager

    with pytest.raises(PDFProcessingError, match="No text could be extracted from the PDF."):
        extract_text_from_pdf("dummy/path/to/image_based.pdf")


@patch('api_main.utils.pdf_helper.pdfplumber.open', side_effect=Exception("Corrupted file"))
def test_extract_text_from_pdf_handles_library_exception(_mock_pdfplumber_open):
    """Tests that general exceptions from pdfplumber are re-raised as PDFProcessingError."""
    with pytest.raises(PDFProcessingError, match="Failed to extract text from PDF: Corrupted file"):
        extract_text_from_pdf("dummy/path/to/corrupted.pdf")


@patch('api_main.utils.pdf_helper.simple_chunker')
@patch('api_main.utils.pdf_helper.extract_text_from_pdf')
def test_process_pdf_orchestration(mock_extract_text, mock_simple_chunker):
    """Tests that process_pdf correctly calls its extractor and chunker dependencies."""
    mock_extract_text.return_value = "Full extracted text."
    mock_simple_chunker.return_value = ["chunk1", "chunk2"]

    result_chunks = process_pdf("dummy/path.pdf")

    mock_extract_text.assert_called_once_with("dummy/path.pdf")
    mock_simple_chunker.assert_called_once_with(
        "Full extracted text.",
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    assert result_chunks == ["chunk1", "chunk2"]


@patch('api_main.utils.pdf_helper.extract_text_from_pdf', side_effect=PDFProcessingError("Test Error"))
def test_process_pdf_propagates_extraction_error(_mock_extract_text):
    """Tests that process_pdf correctly propagates exceptions from its dependencies."""
    with pytest.raises(PDFProcessingError, match="Test Error"):
        process_pdf("dummy/failing.pdf")
