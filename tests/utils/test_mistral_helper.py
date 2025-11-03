from unittest.mock import patch, MagicMock

import pytest

from api_main.utils import mistral_helper


@pytest.fixture(autouse=True)
def mock_mistral_client():
    """Fixture to automatically mock the Mistral client for all tests in this file."""
    with patch('api_main.utils.mistral_helper.client', new_callable=MagicMock) as mock_client:
        yield mock_client


def test_get_embeddings_from_str_list(mock_mistral_client):
    """Tests the embedding creation for a list of strings."""
    mock_embedding_data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_mistral_client.embeddings.create.return_value = MagicMock(data=mock_embedding_data)
    embeddings = mistral_helper.get_embeddings_from_str_list(["text1", "text2"])
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_get_embedding_from_str(mock_mistral_client):
    """Tests the embedding creation for a single string."""
    mock_embedding_data = [MagicMock(embedding=[0.5, 0.6])]
    mock_mistral_client.embeddings.create.return_value = MagicMock(data=mock_embedding_data)
    embedding = mistral_helper.get_embedding_from_str("query text")
    assert embedding == [0.5, 0.6]


def test_get_embeddings_api_error_returns_empty(mock_mistral_client):
    """Tests that an empty list is returned if the API call fails."""
    mock_mistral_client.embeddings.create.side_effect = Exception("API Down")
    assert mistral_helper.get_embeddings_from_str_list(["some text"]) == []
    assert mistral_helper.get_embedding_from_str("some text") == []


def test_merge_and_rerank():
    """Tests the merging and re-ranking logic for search results."""
    vector_res = [
        {"text_chunk": "apple is a fruit", "score": 0.9, "source_file_name": "doc1.pdf"},
        {"text_chunk": "banana is yellow", "score": 0.7, "source_file_name": "doc2.pdf"}
    ]
    keyword_res = [
        {"text_chunk": "apple is a fruit", "score": 0.8, "source_file_name": "doc1.pdf"},
        {"text_chunk": "orange is a citrus", "score": 0.85, "source_file_name": "doc3.pdf"}
    ]
    reranked = mistral_helper.merge_and_rerank(vector_res, keyword_res)
    expected = [
        {"text_chunk": "apple is a fruit", "score": 0.9, "source_file_name": "doc1.pdf"},
        {"text_chunk": "orange is a citrus", "score": 0.85, "source_file_name": "doc3.pdf"},
        {"text_chunk": "banana is yellow", "score": 0.7, "source_file_name": "doc2.pdf"}
    ]
    assert reranked == expected


@patch('api_main.utils.mistral_helper._execute_llm_call')
def test_llm_fact_check_passes_when_llm_returns_true(mock_execute_llm):
    """Tests that fact check passes if the LLM fact-checker returns 'true'."""
    mock_execute_llm.return_value = "true"
    assert mistral_helper._llm_fact_check("some claim", "context", []) is True


@patch('api_main.utils.mistral_helper._execute_llm_call')
def test_llm_fact_check_fails_when_llm_returns_false(mock_execute_llm):
    """Tests that fact check fails if the LLM fact-checker returns 'false'."""
    mock_execute_llm.return_value = "false"
    assert mistral_helper._llm_fact_check("some claim", "context", []) is False


@pytest.mark.parametrize("canned_response", [
    "I could not find the answer", "Insufficient evidence", "I cannot answer"
])
def test_llm_fact_check_passes_for_canned_responses(canned_response):
    """Tests that fact-checking is skipped for predefined refusal answers."""
    assert mistral_helper._llm_fact_check(canned_response, "context", []) is True


@patch('api_main.utils.mistral_helper._execute_llm_call', side_effect=Exception("API Error"))
def test_llm_fact_check_fails_on_api_error(_mock_execute_llm):
    """Tests that fact check fails gracefully if the LLM call has an exception."""
    assert mistral_helper._llm_fact_check("some claim", "context", []) is False


@patch('api_main.utils.mistral_helper._llm_fact_check', return_value=True)
def test_get_llm_response_fact_check_passes(_mock_fact_check, mock_mistral_client):
    """Tests the main response generation path when fact-checking passes."""
    mock_mistral_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Generated Answer"))])
    response = mistral_helper.get_llm_response("query", reranked_results=[], chat_history=[])
    assert response == "Generated Answer"


@patch('api_main.utils.mistral_helper._llm_fact_check', return_value=False)
def test_get_llm_response_fact_check_fails(_mock_fact_check, mock_mistral_client):
    """Tests the path where the answer fails fact-check but there was context."""
    mock_mistral_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hallucinated Answer"))])
    # Provide some reranked_results to avoid the 'Insufficient evidence' path.
    some_context = [{"text_chunk": "some context", "score": 0.8}]
    response = mistral_helper.get_llm_response("query", reranked_results=some_context, chat_history=[])
    assert "could not verify it against the provided information" in response


@patch('api_main.utils.mistral_helper._llm_fact_check', return_value=False)
def test_get_llm_response_insufficient_evidence(_mock_fact_check, mock_mistral_client):
    """Tests returning 'insufficient evidence' when context is missing and fact-check fails."""
    mock_mistral_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Made-up Answer"))])
    # No context is provided, triggering the specific 'insufficient evidence' logic.
    response = mistral_helper.get_llm_response("query", reranked_results=[], chat_history=[])
    assert "Insufficient evidence" in response


@pytest.mark.parametrize("llm_output, expected_intent", [
    ("RAG_QUERY", "RAG_QUERY"),
    ("  CHITCHAT\n", "CHITCHAT"),
    ("REFUSAL.", "REFUSAL"),
    ("UNKNOWN_INTENT", "RAG_QUERY")
])
def test_get_query_intent(mock_mistral_client, llm_output, expected_intent):
    """Tests intent classification for various LLM outputs."""
    mock_mistral_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=llm_output))])
    intent = mistral_helper.get_query_intent("user query", chat_history=[])
    assert intent == expected_intent


def test_get_query_intent_handles_api_error(mock_mistral_client):
    """Tests that intent classification falls back to RAG_QUERY on API error."""
    mock_mistral_client.chat.complete.side_effect = Exception("API Down")
    intent = mistral_helper.get_query_intent("user query", chat_history=[])
    assert intent == "RAG_QUERY"


def test_get_chitchat_response(mock_mistral_client):
    """Tests the chitchat response function."""
    mock_mistral_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello there!"))])
    response = mistral_helper.get_chitchat_response("Hi", chat_history=[])
    assert response == "Hello there!"
    call_args = mock_mistral_client.chat.complete.call_args
    assert call_args.kwargs['temperature'] == 0.5
