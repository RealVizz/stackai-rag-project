from unittest.mock import patch, MagicMock

import pytest

from api_main.services import rag_service
from api_main.utils.pdf_helper import PDFProcessingError


@pytest.fixture
def mock_file():
    """Fixture to create a mock UploadFile object."""
    mock = MagicMock()
    mock.filename = "test.pdf"
    mock.file = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def mock_helpers():
    """Fixture to mock all helper modules used by the RAG service."""
    # Patching where the objects are defined (e.g., pathlib) or where they are
    # imported and used (e.g., shutil in rag_service).
    with (
        patch('api_main.services.rag_service.process_pdf') as mock_process_pdf,
        patch('api_main.services.rag_service.get_embeddings_from_str_list') as mock_get_embeddings,
        patch('api_main.services.rag_service.add_embeddings') as mock_add_embeddings,
        patch('api_main.services.rag_service.add_chunks_to_kw_db_index') as mock_add_kw_index,
        patch('api_main.services.rag_service.get_chat_history') as mock_get_history,
        patch('api_main.services.rag_service.get_query_intent') as mock_get_intent,
        patch('api_main.services.rag_service.get_embedding_from_str') as mock_get_query_embedding,
        patch('api_main.services.rag_service.get_vector_store_chat_data') as mock_get_vector_data,
        patch('api_main.services.rag_service.get_top_k_vector_results') as mock_vector_search,
        patch('api_main.services.rag_service.search_keywords') as mock_keyword_search,
        patch('api_main.services.rag_service.merge_and_rerank') as mock_rerank,
        patch('api_main.services.rag_service.get_llm_response') as mock_get_llm_response,
        patch('api_main.services.rag_service.get_chitchat_response') as mock_get_chitchat,
        patch('api_main.services.rag_service.add_chat_turn') as mock_add_turn,
        patch('api_main.services.rag_service.shutil.copyfileobj'),
        # Patching the Path object in its original library.
        patch('pathlib.Path.mkdir'),
        patch('pathlib.Path.open'),
        patch('pathlib.Path.exists', return_value=True),
        patch('pathlib.Path.unlink') as mock_unlink
    ):
        yield {
            "process_pdf": mock_process_pdf,
            "get_embeddings": mock_get_embeddings,
            "add_embeddings": mock_add_embeddings,
            "add_kw_index": mock_add_kw_index,
            "get_history": mock_get_history,
            "get_intent": mock_get_intent,
            "get_query_embedding": mock_get_query_embedding,
            "get_vector_data": mock_get_vector_data,
            "vector_search": mock_vector_search,
            "keyword_search": mock_keyword_search,
            "rerank": mock_rerank,
            "get_llm_response": mock_get_llm_response,
            "get_chitchat": mock_get_chitchat,
            "add_turn": mock_add_turn,
            "unlink": mock_unlink
        }


def test_process_pdf_upload_success(mock_helpers, mock_file):
    """Tests the successful processing of a single PDF file upload."""
    mock_helpers["process_pdf"].return_value = ["chunk1"]
    mock_helpers["get_embeddings"].return_value = [[1.0]]
    mock_helpers["add_embeddings"].return_value = [{"chunk_id": "id1"}]
    results = rag_service.process_pdf_upload("user1", "chat1", [mock_file])
    assert results[0]["status"] == "success"
    mock_helpers["process_pdf"].assert_called_once()
    mock_helpers["get_embeddings"].assert_called_once()
    mock_helpers["add_embeddings"].assert_called_once()
    mock_helpers["add_kw_index"].assert_called_once()


def test_process_pdf_upload_unsupported_file_type(mock_file):
    """Tests that a non-PDF file is rejected as expected."""
    mock_file.filename = "test.txt"
    results = rag_service.process_pdf_upload("user1", "chat1", [mock_file])
    assert results[0]["status"] == "error"
    assert "Unsupported file type" in results[0]["message"]


def test_process_pdf_upload_processing_error(mock_helpers, mock_file):
    """Tests error handling when PDF processing fails and that cleanup is called."""
    mock_helpers["process_pdf"].side_effect = PDFProcessingError("Test error")
    results = rag_service.process_pdf_upload("user1", "chat1", [mock_file])
    assert results[0]["status"] == "error"
    assert "Test error" in results[0]["message"]
    mock_helpers["unlink"].assert_called_once()


def test_process_pdf_upload_embedding_failure(mock_helpers, mock_file):
    """Tests error handling when the embedding service fails."""
    mock_helpers["process_pdf"].return_value = ["chunk1"]
    mock_helpers["get_embeddings"].return_value = []  # Simulate embedding failure.
    results = rag_service.process_pdf_upload("user1", "chat1", [mock_file])
    assert results[0]["status"] == "error"
    assert "Failed to generate embeddings" in results[0]["message"]
    mock_helpers["unlink"].assert_called_once()


def test_handle_rag_query_orchestration(mock_helpers):
    """Tests the orchestration of the main RAG pipeline."""
    # This is a test of the private function _handle_rag_query, called by process_query.
    rag_service._handle_rag_query("user1", "chat1", "query", [])
    mock_helpers["get_query_embedding"].assert_called_once_with("query")
    mock_helpers["get_vector_data"].assert_called_once_with("user1", "chat1")
    mock_helpers["vector_search"].assert_called_once()
    mock_helpers["keyword_search"].assert_called_once()
    mock_helpers["rerank"].assert_called_once()
    mock_helpers["get_llm_response"].assert_called_once()


def test_process_query_handles_rag_query(mock_helpers):
    """Tests that a RAG_QUERY intent correctly calls the RAG handler."""
    mock_helpers["get_intent"].return_value = "RAG_QUERY"
    # We replace the implementation of _handle_rag_query for this test.
    with patch('api_main.services.rag_service._handle_rag_query') as mock_rag_handler:
        rag_service.process_query("user1", "chat1", "What is RAG?")
        mock_rag_handler.assert_called_once()


def test_process_query_handles_chitchat_query(mock_helpers):
    """Tests that a CHITCHAT intent correctly calls the chitchat handler."""
    mock_helpers["get_intent"].return_value = "CHITCHAT"
    rag_service.process_query("user1", "chat1", "Hello there")
    mock_helpers["get_chitchat"].assert_called_once()
    mock_helpers["add_turn"].assert_called_once()


def test_process_query_handles_refusal_query(mock_helpers):
    """Tests that a REFUSAL intent returns the canned refusal message."""
    mock_helpers["get_intent"].return_value = "REFUSAL"
    response = rag_service.process_query("user1", "chat1", "Harmful query")
    assert "I cannot answer" in response
    mock_helpers["add_turn"].assert_called_once()


def test_process_query_handles_unexpected_intent(mock_helpers):
    """Tests that an unexpected intent defaults to the RAG handler for safety."""
    mock_helpers["get_intent"].return_value = "BOGUS_INTENT"
    # We replace the implementation of _handle_rag_query for this test.
    with patch('api_main.services.rag_service._handle_rag_query') as mock_rag_handler:
        rag_service.process_query("user1", "chat1", "Some query")
        mock_rag_handler.assert_called_once()
