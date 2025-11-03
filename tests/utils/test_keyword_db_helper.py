import json

import pytest

from api_main.utils import keyword_db_helper

# Sample data for chunks to be indexed
SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk1",
        "text_chunk": "The quick brown fox jumps over the lazy dog.",
        "source_file_name": "doc1.pdf"
    },
    {
        "chunk_id": "chunk2",
        "text_chunk": "A lazy fox is not a quick fox.",
        "source_file_name": "doc2.pdf"
    },
    {
        "chunk_id": "chunk3",
        "text_chunk": "The dog is happy.",
        "source_file_name": "doc1.pdf"
    }
]


@pytest.fixture
def mock_db_file(tmp_path):
    """Pytest fixture to create a temporary keyword DB file for testing."""
    return tmp_path / "test_keyword_db.json"


@pytest.fixture
def mock_stop_words_file(tmp_path):
    """Pytest fixture to create a temporary stop words file."""
    file = tmp_path / "stop_words.txt"
    file.write_text("the\na\nis\nover")
    return file


@pytest.fixture(autouse=True)
def isolate_keyword_store(monkeypatch, mock_db_file, mock_stop_words_file):
    """Fixture to isolate the keyword store, file paths, and stop words for each test."""
    monkeypatch.setattr(keyword_db_helper, "KEYWORD_STORE", {})
    monkeypatch.setattr(keyword_db_helper, "KEYWORD_DB_FILE_PATH", str(mock_db_file))
    monkeypatch.setattr(keyword_db_helper, "_STOP_WORDS", keyword_db_helper._load_stop_words(
        mock_stop_words_file))


def test_load_keyword_db_from_persistent_storage_success(mock_db_file):
    """Tests that the keyword DB is loaded correctly from a valid JSON file."""
    sample_db = {"user1": {"chat1": {"inverted_index": {"fox": ["chunk1"]}}}}
    mock_db_file.write_text(json.dumps(sample_db))
    keyword_db_helper.load_keyword_db_from_persistent_storage()
    assert keyword_db_helper.KEYWORD_STORE == sample_db


def test_load_keyword_db_from_persistent_storage_corrupted_file(mock_db_file):
    """Tests that loading handles a corrupted or invalid JSON DB file gracefully."""
    mock_db_file.write_text("this is not valid json {")
    # This call should not raise an exception.
    keyword_db_helper.load_keyword_db_from_persistent_storage()
    # The store should be reset to empty instead of crashing.
    assert keyword_db_helper.KEYWORD_STORE == {}


def test_add_chunks_to_kw_db_index(mock_db_file):
    """Tests the entire indexing process for a list of chunks."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    store = keyword_db_helper.KEYWORD_STORE
    chat_data = store["user1"]["chat1"]
    assert sorted(chat_data["inverted_index"]["fox"]) == ["chunk1", "chunk2"]
    assert chat_data["chunk_metadata"]["chunk1"]["tf"]["quick"] == 1
    assert chat_data["total_chunks_in_chat"] == 3
    assert json.loads(mock_db_file.read_text()) == store


def test_add_chunks_is_idempotent():
    """Tests that adding the same chunks multiple times does not create duplicates."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    first_run_store = json.loads(json.dumps(keyword_db_helper.KEYWORD_STORE))
    first_total_chunks = first_run_store["user1"]["chat1"]["total_chunks_in_chat"]

    # Add the same chunks again.
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    second_run_store = keyword_db_helper.KEYWORD_STORE
    second_total_chunks = second_run_store["user1"]["chat1"]["total_chunks_in_chat"]

    assert first_run_store == second_run_store
    assert first_total_chunks == second_total_chunks


def test_search_keywords_single_term():
    """Tests searching for a single term that exists in multiple documents."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    results = keyword_db_helper.search_keywords("user1", "chat1", "fox", SAMPLE_CHUNKS)
    assert len(results) == 2
    found_texts = {res["text_chunk"] for res in results}
    assert SAMPLE_CHUNKS[0]["text_chunk"] in found_texts
    assert SAMPLE_CHUNKS[1]["text_chunk"] in found_texts


def test_search_keywords_multiple_terms():
    """Tests searching for multiple terms that require intersection."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    results = keyword_db_helper.search_keywords("user1", "chat1", "quick lazy", SAMPLE_CHUNKS)
    assert len(results) == 2
    found_texts = {res["text_chunk"] for res in results}
    assert SAMPLE_CHUNKS[0]["text_chunk"] in found_texts
    assert SAMPLE_CHUNKS[1]["text_chunk"] in found_texts


def test_search_keywords_no_match():
    """Tests that an empty list is returned for a term that doesn't exist."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    results = keyword_db_helper.search_keywords("user1", "chat1", "nonexistent_term", SAMPLE_CHUNKS)
    assert results == []


def test_search_keywords_with_stop_words_only():
    """Tests that a query with only stop words returns no results."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    results = keyword_db_helper.search_keywords("user1", "chat1", "the a is", SAMPLE_CHUNKS)
    assert results == []


def test_search_keywords_empty_query():
    """Tests that an empty query string returns no results."""
    keyword_db_helper.add_chunks_to_kw_db_index("user1", "chat1", SAMPLE_CHUNKS)
    results = keyword_db_helper.search_keywords("user1", "chat1", "", SAMPLE_CHUNKS)
    assert results == []


def test_search_is_resilient_to_malformed_store():
    """Tests that search returns an empty list if the store is corrupted."""
    # Simulate a corrupted store where the inverted_index is not a dictionary.
    corrupted_store = {
        "user1": {
            "chat1": {
                "inverted_index": ["this should be a dict"],
                "chunk_metadata": {},
                "total_chunks_in_chat": 0
            }
        }
    }
    keyword_db_helper.KEYWORD_STORE = corrupted_store

    # This search should not crash; it should fail gracefully.
    results = keyword_db_helper.search_keywords("user1", "chat1", "any query", SAMPLE_CHUNKS)
    assert results == []
