import json

import numpy as np
import pytest

from api_main.utils import vector_db_helper
from api_main.utils.vector_db_helper import SimilarityMetric

# Sample data for embeddings
SAMPLE_CHUNKS = ["the cat sat on the mat", "a dog chased the cat"]
SAMPLE_EMBEDDINGS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.fixture
def mock_db_file(tmp_path):
    """Pytest fixture to create a temporary vector DB file for testing."""
    return tmp_path / "test_vector_db.json"


@pytest.fixture(autouse=True)
def isolate_vector_store(monkeypatch, mock_db_file):
    """Fixture to isolate the vector store and file path for each test."""
    monkeypatch.setattr(vector_db_helper, "VECTOR_STORE", {})
    monkeypatch.setattr(vector_db_helper, "VECTOR_DB_FILE_PATH", str(mock_db_file))


def test_load_vector_db_from_persistent_storage_success(mock_db_file):
    """Tests that the vector DB is loaded correctly from a valid JSON file."""
    sample_db = {"user1": {"chat1": [{"chunk_id": "id1"}]}}
    mock_db_file.write_text(json.dumps(sample_db))
    vector_db_helper.load_vector_db_from_persistent_storage()
    assert vector_db_helper.VECTOR_STORE == sample_db


def test_load_vector_db_from_persistent_storage_corrupted_file(mock_db_file):
    """Tests that loading handles a corrupted or invalid JSON DB file gracefully."""
    mock_db_file.write_text("this is not valid json {")
    vector_db_helper.load_vector_db_from_persistent_storage()
    assert vector_db_helper.VECTOR_STORE == {}


def test_add_embeddings(mock_db_file):
    """Tests the process of adding new chunks and their embeddings."""
    new_entries = vector_db_helper.add_embeddings(
        "user1", "chat1", "uuid1", "doc1.pdf", SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS
    )
    assert len(new_entries) == 2
    store = vector_db_helper.VECTOR_STORE
    assert len(store["user1"]["chat1"]) == 2
    assert json.loads(mock_db_file.read_text()) == store


def test_add_embeddings_handles_malformed_data():
    """Tests that add_embeddings safely overwrites malformed data for a user."""
    vector_db_helper.VECTOR_STORE = {"user1": "this should be a dict of chats"}
    # This call should not crash; it should fix the entry for "user1".
    vector_db_helper.add_embeddings("user1", "chat1", "uuid1", "doc1.pdf", ["chunk"], [[1.0]])
    assert isinstance(vector_db_helper.VECTOR_STORE["user1"], dict)
    assert len(vector_db_helper.VECTOR_STORE["user1"]["chat1"]) == 1


def test_get_vector_store_chat_data():
    """Tests retrieval of all vector data for a specific chat."""
    vector_db_helper.add_embeddings("user1", "chat1", "uuid1", "doc1.pdf", ["chunk"], [[1.0]])
    data = vector_db_helper.get_vector_store_chat_data("user1", "chat1")
    assert len(data) == 1
    assert vector_db_helper.get_vector_store_chat_data("user_xxx", "chat_xxx") == []


def test_cosine_similarity():
    """Tests the cosine similarity calculation with known vectors."""
    assert vector_db_helper._cosine_similarity(np.array([1, 0]), np.array([0, 1])) == 0.0
    assert np.isclose(vector_db_helper._cosine_similarity(np.array([1, 1]), np.array([1, 1])), 1.0)
    assert np.isclose(vector_db_helper._cosine_similarity(np.array([1, 0]), np.array([-1, 0])), -1.0)
    assert vector_db_helper._cosine_similarity(np.array([0, 0]), np.array([1, 1])) == 0.0


def test_get_top_k_vector_results_cosine():
    """Tests the main vector search functionality with cosine similarity."""
    chat_data = [
        {"embedding": [1.0, 0.0, 0.0], "text_chunk": "A", "source_file_name": "f1"},
        {"embedding": [0.1, 0.9, 0.1], "text_chunk": "B", "source_file_name": "f2"},
        {"embedding": [0.8, 0.2, 0.0], "text_chunk": "C", "source_file_name": "f3"}
    ]
    query_embedding = [1.0, 0.0, 0.0]
    results = vector_db_helper.get_top_k_vector_results(chat_data, query_embedding, k=2, threshold=0.5)
    assert len(results) == 2
    assert results[0]["text_chunk"] == "A"
    assert results[1]["text_chunk"] == "C"


def test_get_top_k_removes_duplicates():
    """Tests that the search logic removes duplicate text chunks before scoring."""
    chat_data = [
        {"embedding": [1.0, 0.0], "text_chunk": "duplicate text", "source_file_name": "f1"},
        {"embedding": [0.9, 0.1], "text_chunk": "duplicate text", "source_file_name": "f2"},
        {"embedding": [0.0, 1.0], "text_chunk": "unique text", "source_file_name": "f3"}
    ]
    query_embedding = [1.0, 0.0]
    results = vector_db_helper.get_top_k_vector_results(chat_data, query_embedding, k=2, threshold=0.5)
    assert len(results) == 1
    assert results[0]["text_chunk"] == "duplicate text"


def test_get_top_k_vector_results_euclidean():
    """Tests the main vector search functionality with Euclidean-based similarity."""
    chat_data = [
        {"embedding": [1.0, 0.0], "text_chunk": "A", "source_file_name": "f1"},
        {"embedding": [5.0, 5.0], "text_chunk": "B", "source_file_name": "f2"},
        {"embedding": [1.0, 2.0], "text_chunk": "C", "source_file_name": "f3"}
    ]
    query_embedding = [0.0, 0.0]
    results = vector_db_helper.get_top_k_vector_results(
        chat_data, query_embedding, k=3, threshold=0.1, metric=SimilarityMetric.EUCLIDEAN
    )
    assert len(results) == 3
    assert results[0]["text_chunk"] == "A"
    assert results[1]["text_chunk"] == "C"
    assert results[2]["text_chunk"] == "B"


def test_get_top_k_handles_empty_input():
    """Tests that the search function handles empty inputs gracefully."""
    assert vector_db_helper.get_top_k_vector_results([], [1.0]) == []


def test_calculate_similarity_unknown_metric_raises_error():
    """Tests that an unknown similarity metric raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown similarity metric: unknown"):
        vector_db_helper._calculate_similarity([1.0], [1.0], "unknown")
