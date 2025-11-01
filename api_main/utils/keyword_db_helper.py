import json
import os
import re
import threading
from pathlib import Path

from config.config import KEYWORD_DB_FILE_PATH, STOP_WORDS_FILE_PATH

_db_lock = threading.Lock()


def _load_stop_words(file_path: Path) -> set:
    if not file_path.exists():
        print(f"Warning: Stop words file not found at {file_path}. Using empty set.")
        return set()
    try:
        with open(file_path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Error loading stop words file: {e}. Using empty set.")
        return set()


_STOP_WORDS = _load_stop_words(STOP_WORDS_FILE_PATH)

# Structure: { "user_id": { "chat_id": { "inverted_index": {"keyword": [chunk_id1, chunk_id2]} } } }
KEYWORD_STORE: dict[str, dict[str, dict[str, dict]]] = {}


def _save_to_disk():
    """Saving the entire in-memory KEYWORD_STORE to a JSON file."""
    try:
        db_path = Path(KEYWORD_DB_FILE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w") as f:
            json.dump(KEYWORD_STORE, f, indent=2)
    except Exception as e:
        print(f"Error saving keyword store to disk: {e}")


def _tokenize_text(text: str):
    """Removes punctuation, lowercases, and filters stop words."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in _STOP_WORDS and len(word) > 2]


def _update_index_for_chunks(inverted_index: dict, chunks: list[dict]):
    for chunk_data in chunks:
        chunk_id = chunk_data.get("chunk_id")
        text_chunk = chunk_data.get("text_chunk")

        if not chunk_id or not text_chunk:
            continue

        tokens = _tokenize_text(text_chunk)
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            if chunk_id not in inverted_index[token]:
                inverted_index[token].append(chunk_id)
    return inverted_index


def _get_chat_index(user_id: str, chat_id: str):
    with _db_lock:
        return KEYWORD_STORE.get(user_id, {}).get(chat_id, {}).get("inverted_index", {})


def _find_matching_chunk_ids(chat_index: dict, query_tokens: list[str]):
    if not query_tokens or not chat_index:
        return set()

    try:
        matching_chunk_ids = set(chat_index.get(query_tokens[0], []))

        # [optimization] If no matches for the first token, or only one token, we're done.
        if not matching_chunk_ids or len(query_tokens) == 1:
            return matching_chunk_ids

        for token in query_tokens[1:]:
            matching_chunk_ids.intersection_update(chat_index.get(token, []))
            if not matching_chunk_ids:  # If intersection is empty, we can stop early.
                return set()

        return matching_chunk_ids
    except Exception as e:
        print(f"Error during keyword search intersection: {e}")
        return set()


def _build_results_from_chunk_ids(all_chat_chunks: list[dict], chunk_ids: set):
    if not chunk_ids:
        return []

    # Creating a quick-lookup map for all chunks.
    all_chunks_map = {chunk["chunk_id"]: chunk for chunk in all_chat_chunks}

    final_results = []
    for chunk_id in chunk_ids:
        chunk_data = all_chunks_map.get(chunk_id)
        if chunk_data:
            # Using the same format as the vector search for easy merging.
            final_results.append({
                "score": 1.0,  # A default high score for keyword matches (for now).
                "source_file_name": chunk_data["source_file_name"],
                "text_chunk": chunk_data["text_chunk"]
            })

    return final_results


def load_keyword_db_from_persistent_storage():  #load_keyword_db_from_persistent_storage
    global KEYWORD_STORE
    with _db_lock:
        if os.path.exists(KEYWORD_DB_FILE_PATH):
            try:
                with open(KEYWORD_DB_FILE_PATH, "r") as f:
                    KEYWORD_STORE = json.load(f)
            except Exception as e:
                print(f"Error loading keyword store from disk: {e}")
                KEYWORD_STORE = {}
        else:
            print("No persistent keyword DB file found. Starting fresh.")
            KEYWORD_STORE = {}


def add_chunks_to_kw_db_index(user_id: str, chat_id: str, chunks: list[dict]):
    global KEYWORD_STORE
    with _db_lock:
        if user_id not in KEYWORD_STORE:
            KEYWORD_STORE[user_id] = {}
        if chat_id not in KEYWORD_STORE[user_id]:
            KEYWORD_STORE[user_id][chat_id] = {"inverted_index": {}}

        # Get the current index
        inverted_index = KEYWORD_STORE[user_id][chat_id].get("inverted_index", {})

        # Update the index with new chunks
        updated_index = _update_index_for_chunks(inverted_index, chunks)

        KEYWORD_STORE[user_id][chat_id]["inverted_index"] = updated_index
        _save_to_disk()


def search_keywords(user_id: str, chat_id: str, query_str: str, all_chat_chunks: list[dict]):
    chat_index = _get_chat_index(user_id, chat_id)
    if not chat_index:
        return []

    query_tokens = _tokenize_text(query_str)
    if not query_tokens:
        return []

    matching_chunk_ids = _find_matching_chunk_ids(chat_index, query_tokens)
    if not matching_chunk_ids:
        return []

    return _build_results_from_chunk_ids(all_chat_chunks, matching_chunk_ids)
