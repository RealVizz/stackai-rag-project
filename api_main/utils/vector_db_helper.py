import json
import os
import uuid
import threading
from config.config import VECTOR_DB_FILE_PATH
from pathlib import Path

import numpy as np
from enum import Enum

class SimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

_db_lock = threading.Lock()

VECTOR_STORE: dict[str, dict[str, list[dict]]] = {}

def _save_to_disk():
    """
    Note : It uses a simple "overwrite" strategy.
    IMPORTANT: This function assumes the caller already holds _db_lock.
    """
    try:
        db_path = Path(VECTOR_DB_FILE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(db_path, "w") as f:
            json.dump(VECTOR_STORE, f, indent=2)

    except Exception as e:
        print(f"Error saving to disk: {e}")


def load_vector_db_from_persistent_storage():
    """
    This is called on server startup, is thread-safe.
    """
    global VECTOR_STORE

    with _db_lock:
        if os.path.exists(VECTOR_DB_FILE_PATH):
            try:
                with open(VECTOR_DB_FILE_PATH, "r") as f:
                    VECTOR_STORE = json.load(f)

            except Exception as e:
                print(f"Error loading from disk: {e}")
                VECTOR_STORE = {}
        else:
            print("No persistent DB file found. Starting with empty vector store.")
            VECTOR_STORE = {}


def add_embeddings(user_id: str, chat_id: str, source_file_uuid: str, source_file_name: str,
                   chunks: list[str], embeddings: list[list[float]]):
    global VECTOR_STORE

    with _db_lock:  # Acquiring the lock for the entire operation
        if user_id not in VECTOR_STORE:
            VECTOR_STORE[user_id] = {}
        if chat_id not in VECTOR_STORE[user_id]:
            VECTOR_STORE[user_id][chat_id] = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            embedding = embeddings[i]

            data_entry = {
                "chunk_id": chunk_id,
                "source_file_uuid": source_file_uuid,
                "source_file_name": source_file_name,
                "text_chunk": chunk,
                "embedding": embedding
            }
            VECTOR_STORE[user_id][chat_id].append(data_entry)

        _save_to_disk()

    return None


def get_vector_store_chat_data(user_id: str, chat_id: str):
    with _db_lock:
        vector_store_chat_data = VECTOR_STORE.get(user_id, {}).get(chat_id, [])
        return list(vector_store_chat_data)

def _get_unique_chunks_from(all_chunks: list[dict]):
    unique_chunks_map = {}
    for chunk_data in all_chunks:
        cleaner_text = chunk_data["text_chunk"].lower().strip()
        if cleaner_text not in unique_chunks_map:
            unique_chunks_map[cleaner_text] = chunk_data

    return list(unique_chunks_map.values())


def _sort_and_get_top_k(relevant_chunks: list[dict], k: int):
    relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
    return relevant_chunks[:k]


def _calculate_similarity(vec1: list[float], vec2: list[float], metric: SimilarityMetric):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    if metric == SimilarityMetric.COSINE:
        return _cosine_similarity(v1, v2)
    elif metric == SimilarityMetric.EUCLIDEAN:
        distance = _euclidean_distance(v1, v2)
        return 1.0 / (1.0 + distance)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:  # Avoid division by zero if either vector is all zeros.
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def _euclidean_distance(v1: np.ndarray, v2: np.ndarray):
    return np.linalg.norm(v1 - v2)


def _calculate_similarity_scores(
        chunks_list: list[dict],
        query_embedding: list[float],
        metric: SimilarityMetric,
        threshold: float):

    relevant_chunks = []
    for chunk_data in chunks_list:
        score = _calculate_similarity(
            query_embedding,
            chunk_data["embedding"],
            metric
        )

        if score >= threshold:
            relevant_chunks.append({
                "score": score,
                "source_file_name": chunk_data["source_file_name"],
                "text_chunk": chunk_data["text_chunk"]
            })
    return relevant_chunks


def get_top_k_vector_results(
        user_id: str,
        chat_id: str,
        query_embedding: list[float],
        k: int = 5,
        threshold: float = 0.5,
        metric: SimilarityMetric = SimilarityMetric.COSINE):

    chat_unique_vector_db_data = get_vector_store_chat_data(user_id, chat_id)
    if not chat_unique_vector_db_data:
        return []

    unique_chunks = _get_unique_chunks_from(chat_unique_vector_db_data)

    relevant_chunks = _calculate_similarity_scores(
        unique_chunks,
        query_embedding,
        metric,
        threshold
    )

    top_k_results = _sort_and_get_top_k(relevant_chunks, k)

    return top_k_results
