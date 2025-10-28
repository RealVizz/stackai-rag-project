import json
import os
import uuid
import threading
from config.config import VECTOR_DB_FILE_PATH
from pathlib import Path

_db_lock = threading.Lock()

# --- In-Memory Cache --- { "user_id": { "chat_id": [ {chunk}, {chunk}, ... ] } }
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

# --- Public Functions ---

def load_from_persistent_storage():
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


def add_embeddings(user_id: str, chat_id: str, source_file: str, chunks: list[str], embeddings: list[list[float]]):
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
                "source_file": source_file,
                "text_chunk": chunk,
                "embedding": embedding
            }
            VECTOR_STORE[user_id][chat_id].append(data_entry)

        _save_to_disk()  # After updating in-memory, save to disk ""while still holding the lock""

    return None


def get_vector_store_chat_data(user_id: str, chat_id: str) -> list[dict]:
    with _db_lock:
        vector_store_chat_data = VECTOR_STORE.get(user_id, {}).get(chat_id, [])
        return list(vector_store_chat_data)

