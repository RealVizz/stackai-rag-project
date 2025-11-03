import json

import pytest

from api_main.utils import chat_memory_helper

# Sample data to be used across tests
SAMPLE_CHAT_HISTORY = {
    "user_123": {
        "chat_abc": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "chat_xyz": [
            {"role": "user", "content": "What is RAG?"}
        ]
    },
    "user_456": {
        "chat_def": [
            {"role": "user", "content": "Thanks"}
        ]
    }
}


@pytest.fixture
def mock_chat_file(tmp_path):
    """Pytest fixture to create a temporary chat history file for testing."""
    return tmp_path / "test_chat_history.json"


@pytest.fixture(autouse=True)
def isolate_chat_history(monkeypatch, mock_chat_file):
    """Fixture to isolate the global CHAT_HISTORY_STORE and file path for each test."""
    monkeypatch.setattr(chat_memory_helper, "CHAT_HISTORY_STORE", {})
    monkeypatch.setattr(chat_memory_helper, "CHAT_HISTORY_FILE_PATH", str(mock_chat_file))


def test_load_chat_from_persistent_storage_file_not_found(mock_chat_file):
    """Tests that loading works gracefully when the history file doesn't exist."""
    assert not mock_chat_file.exists()
    chat_memory_helper.load_chat_from_persistent_storage()
    assert chat_memory_helper.CHAT_HISTORY_STORE == {}


def test_load_chat_from_persistent_storage_success(mock_chat_file):
    """Tests that chat history is loaded correctly from an existing JSON file."""
    mock_chat_file.write_text(json.dumps(SAMPLE_CHAT_HISTORY))
    chat_memory_helper.load_chat_from_persistent_storage()
    assert chat_memory_helper.CHAT_HISTORY_STORE == SAMPLE_CHAT_HISTORY


def test_load_chat_from_persistent_storage_corrupted_file(mock_chat_file):
    """Tests that loading handles corrupted or invalid JSON gracefully."""
    mock_chat_file.write_text("this is not valid json")
    chat_memory_helper.load_chat_from_persistent_storage()
    assert chat_memory_helper.CHAT_HISTORY_STORE == {}


def test_get_chat_history():
    """Tests retrieval of chat history for various cases."""
    chat_memory_helper.CHAT_HISTORY_STORE = SAMPLE_CHAT_HISTORY

    history = chat_memory_helper.get_chat_history("user_123", "chat_abc")
    assert history == SAMPLE_CHAT_HISTORY["user_123"]["chat_abc"]

    history_nonexistent_chat = chat_memory_helper.get_chat_history("user_123", "chat_nonexistent")
    assert history_nonexistent_chat == []

    history_nonexistent_user = chat_memory_helper.get_chat_history("user_nonexistent", "chat_abc")
    assert history_nonexistent_user == []


def test_get_chat_history_returns_a_copy():
    """Tests that get_chat_history returns a copy, not a reference, to protect the store."""
    chat_memory_helper.CHAT_HISTORY_STORE = json.loads(json.dumps(SAMPLE_CHAT_HISTORY))  # Deep copy

    retrieved_history = chat_memory_helper.get_chat_history("user_123", "chat_abc")
    retrieved_history.append({"role": "user", "content": "This should not be in the original"})

    original_history = chat_memory_helper.get_chat_history("user_123", "chat_abc")

    assert len(original_history) == 2
    assert original_history == SAMPLE_CHAT_HISTORY["user_123"]["chat_abc"]


def test_add_chat_turn_and_persistence(mock_chat_file):
    """Tests adding a new chat turn and verifies it's persisted to the file."""
    chat_memory_helper.add_chat_turn("new_user", "new_chat", "First query", "First response")

    expected_first_turn = {
        "new_user": {
            "new_chat": [
                {"role": "user", "content": "First query"},
                {"role": "assistant", "content": "First response"}
            ]
        }
    }
    assert chat_memory_helper.CHAT_HISTORY_STORE == expected_first_turn
    assert json.loads(mock_chat_file.read_text()) == expected_first_turn

    chat_memory_helper.add_chat_turn("new_user", "new_chat", "Second query", "Second response")

    expected_second_turn = {
        "new_user": {
            "new_chat": [
                {"role": "user", "content": "First query"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second query"},
                {"role": "assistant", "content": "Second response"}
            ]
        }
    }
    assert chat_memory_helper.CHAT_HISTORY_STORE == expected_second_turn
    assert json.loads(mock_chat_file.read_text()) == expected_second_turn


def test_add_chat_turn_handles_malformed_user_data():
    """Tests that add_chat_turn safely overwrites malformed data for a user."""
    chat_memory_helper.CHAT_HISTORY_STORE = {"user_to_fix": "this is not a dict"}

    # This call should not crash, it should fix the entry for "user_to_fix".
    chat_memory_helper.add_chat_turn("user_to_fix", "new_chat", "query", "response")

    expected_store = {
        "user_to_fix": {
            "new_chat": [
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": "response"}
            ]
        }
    }
    assert chat_memory_helper.CHAT_HISTORY_STORE == expected_store


def test_get_all_user_ids():
    """Tests retrieval of all unique user IDs."""
    chat_memory_helper.CHAT_HISTORY_STORE = SAMPLE_CHAT_HISTORY
    user_ids = chat_memory_helper.get_all_user_ids()
    assert sorted(user_ids) == sorted(["user_123", "user_456"])

    chat_memory_helper.CHAT_HISTORY_STORE = {}
    user_ids_empty = chat_memory_helper.get_all_user_ids()
    assert user_ids_empty == []


def test_get_all_chat_ids_for_user():
    """Tests retrieval of all chat IDs for a specific user."""
    chat_memory_helper.CHAT_HISTORY_STORE = SAMPLE_CHAT_HISTORY

    chat_ids = chat_memory_helper.get_all_chat_ids_for_user("user_123")
    assert sorted(chat_ids) == sorted(["chat_abc", "chat_xyz"])

    chat_ids_nonexistent = chat_memory_helper.get_all_chat_ids_for_user("user_nonexistent")
    assert chat_ids_nonexistent == []


def test_get_functions_are_resilient_to_malformed_data():
    """Tests that get functions return defaults instead of crashing on malformed data."""
    malformed_store = {
        "user_good": {"chat_good": [{"role": "user", "content": "Hi"}]},
        "user_bad_chats": "this should be a dict of chats",
        "user_bad_history": {"chat_bad": "this should be a list"}
    }
    chat_memory_helper.CHAT_HISTORY_STORE = malformed_store

    # Should not raise an error.
    assert chat_memory_helper.get_chat_history("user_bad_chats", "any_chat") == []
    assert chat_memory_helper.get_chat_history("user_bad_history", "chat_bad") == []
    assert chat_memory_helper.get_all_chat_ids_for_user("user_bad_chats") == []
