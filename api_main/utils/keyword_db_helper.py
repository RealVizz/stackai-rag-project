import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

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

KEYWORD_STORE: dict[str, dict[str, dict[str, Any]]] = {}


def _save_to_disk():
    """Saving the entire in-memory KEYWORD_STORE to a JSON file."""
    try:
        db_path = Path(KEYWORD_DB_FILE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w") as f:
            json.dump(KEYWORD_STORE, f, indent=2)
    except Exception as e:
        print(f"Error saving keyword store to disk: {e}")


def _clean_and_tokenize(text: str) -> list[str]:
    """Removes punctuation, lowercases, and splits into words."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


def _filter_stop_words(tokens: list[str]):
    """Filters out stop words and short words."""
    # Keep words longer than 2 chars.
    return [word for word in tokens if word not in _STOP_WORDS and len(word) > 2]


def _update_chat_keyword_data(chat_keyword_data: dict, chunks: list[dict]):
    """
    Updates the chat_keyword_data (inverted_index and chunk_metadata) for new chunks.
    This function calculates and stores TF (Term Frequency) data.
    """
    inverted_index = chat_keyword_data.get("inverted_index", {})
    chunk_metadata = chat_keyword_data.get("chunk_metadata", {})

    for chunk_data in chunks:
        chunk_id = chunk_data.get("chunk_id")
        text_chunk = chunk_data.get("text_chunk")

        if not chunk_id or not text_chunk or chunk_id in chunk_metadata:
            # Skip if no data or if chunk is already indexed
            continue

        all_tokens = _clean_and_tokenize(text_chunk)
        filtered_tokens = _filter_stop_words(all_tokens)

        if not filtered_tokens:
            continue

        # Calc. and store TF data.
        tf_counts = Counter(filtered_tokens)
        chunk_metadata[chunk_id] = {
            "total_tokens": len(all_tokens),  # Using len. of 'all' tokens for normalization.
            "tf": dict(tf_counts)
        }

        # Updating inv. index .
        for token in tf_counts.keys():
            if token not in inverted_index:
                inverted_index[token] = []
            if chunk_id not in inverted_index[token]:
                inverted_index[token].append(chunk_id)

        # Updating total chunk count.
        chat_keyword_data["total_chunks_in_chat"] = chat_keyword_data.get("total_chunks_in_chat", 0) + 1

    chat_keyword_data["inverted_index"] = inverted_index
    chat_keyword_data["chunk_metadata"] = chunk_metadata
    return chat_keyword_data


def _get_chat_keyword_data(user_id: str, chat_id: str):
    with _db_lock:
        return KEYWORD_STORE.get(user_id, {}).get(chat_id, {})


def _find_matching_chunk_ids(chat_index: dict, query_tokens: list[str]) -> set:
    if not query_tokens or not chat_index:
        return set()

    try:
        matching_chunk_ids = set(chat_index.get(query_tokens[0], []))

        # [optimization] If no matches for the first token, or only one token, we're done.
        if not matching_chunk_ids or len(query_tokens) == 1:
            return matching_chunk_ids

        # Get the intersection of all other tokens
        for token in query_tokens[1:]:
            matching_chunk_ids.intersection_update(chat_index.get(token, []))
            if not matching_chunk_ids:  # If the intersection is empty, we can stop early.
                return set()

        return matching_chunk_ids
    except Exception as e:
        print(f"Error during keyword search intersection: {e}")
        return set()


def _calculate_tfidf_scores(chat_keyword_data: dict, matching_chunk_ids: set, filtered_query_tokens: list[str]):
    scores = {}
    inverted_index = chat_keyword_data.get("inverted_index", {})
    chunk_metadata = chat_keyword_data.get("chunk_metadata", {})
    total_chunks = chat_keyword_data.get("total_chunks_in_chat", 1)  # default 1, to avoid division by zero.

    # Calc. IDF for each query term.
    idf_scores = {}
    for term in filtered_query_tokens:
        num_chunks_with_term = len(inverted_index.get(term, []))
        # Standard IDF formula, adding 1 to denominator to avoid division by zero
        idf_scores[term] = math.log(total_chunks / (1 + num_chunks_with_term))

    # Calc. TF-IDF score for each chunk.
    for chunk_id in matching_chunk_ids:
        chunk_meta = chunk_metadata.get(chunk_id)
        if not chunk_meta:
            continue

        total_score = 0.0
        tf_counts = chunk_meta.get("tf", {})
        total_tokens_in_chunk = chunk_meta.get("total_tokens", 1)  # Avoid division by zero

        for term in filtered_query_tokens:
            if term in tf_counts:
                tf = tf_counts[term] / total_tokens_in_chunk
                total_score += (tf * idf_scores[term])

        scores[chunk_id] = total_score

    return scores


def _build_results_from_chunk_ids(all_chat_chunks: list[dict], chat_keyword_data: dict, matching_chunk_ids: set,
                                  filtered_query_tokens: list[str]):
    if not matching_chunk_ids:
        return []

    chunk_scores = _calculate_tfidf_scores(chat_keyword_data, matching_chunk_ids, filtered_query_tokens)

    all_chunks_map = {chunk["chunk_id"]: chunk for chunk in all_chat_chunks}
    final_results = []

    for chunk_id, score in chunk_scores.items():
        if score == 0:  # skip zero-score results.
            continue

        chunk_data = all_chunks_map.get(chunk_id)
        if chunk_data:
            final_results.append({
                "score": score,
                "source_file_name": chunk_data["source_file_name"],
                "text_chunk": chunk_data["text_chunk"]
            })

    return final_results


def load_keyword_db_from_persistent_storage():
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
            # --- Initializing new structure ---
            KEYWORD_STORE[user_id][chat_id] = {
                "inverted_index": {},
                "chunk_metadata": {},
                "total_chunks_in_chat": 0
            }

        # Get the current store for this chat
        chat_unique_kw_db_data = KEYWORD_STORE[user_id][chat_id]
        updated_chat_keyword_data = _update_chat_keyword_data(chat_unique_kw_db_data, chunks)
        KEYWORD_STORE[user_id][chat_id] = updated_chat_keyword_data

        _save_to_disk()


def search_keywords(user_id: str, chat_id: str, query_str: str, all_chat_chunks: list[dict]):
    chat_keyword_data = _get_chat_keyword_data(user_id, chat_id)
    chat_index = chat_keyword_data.get("inverted_index", {})

    if not chat_index:
        return []

    query_tokens = _clean_and_tokenize(query_str)
    filtered_query_tokens = _filter_stop_words(query_tokens)

    if not filtered_query_tokens:
        return []

    matching_chunk_ids = _find_matching_chunk_ids(chat_index, filtered_query_tokens)
    if not matching_chunk_ids:
        return []

    return _build_results_from_chunk_ids(all_chat_chunks, chat_keyword_data, matching_chunk_ids, filtered_query_tokens)
