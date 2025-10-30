import json
import os
import threading
from config.config import CHAT_HISTORY_FILE_PATH
from pathlib import Path

_chat_lock = threading.Lock()

CHAT_HISTORY_STORE: dict[str, dict[str, list[dict]]] = {}


def _save_chat_to_disk():
    try:
        db_path = Path(CHAT_HISTORY_FILE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w") as f:
            json.dump(CHAT_HISTORY_STORE, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history to disk: {e}")


def load_chat_from_persistent_storage():
    global CHAT_HISTORY_STORE
    with _chat_lock:
        if os.path.exists(CHAT_HISTORY_FILE_PATH):
            try:
                with open(CHAT_HISTORY_FILE_PATH, "r") as f:
                    CHAT_HISTORY_STORE = json.load(f)
            except Exception as e:
                print(f"Error loading chat history from disk: {e}")
                CHAT_HISTORY_STORE = {}
        else:
            print("No persistent chat history file found. Starting empty.")
            CHAT_HISTORY_STORE = {}


def add_chat_turn(user_id: str, chat_id: str, user_query: str, assistant_response: str):
    global CHAT_HISTORY_STORE
    with _chat_lock:
        if user_id not in CHAT_HISTORY_STORE:
            CHAT_HISTORY_STORE[user_id] = {}
        if chat_id not in CHAT_HISTORY_STORE[user_id]:
            CHAT_HISTORY_STORE[user_id][chat_id] = []

        CHAT_HISTORY_STORE[user_id][chat_id].append({"role": "user", "content": user_query})
        CHAT_HISTORY_STORE[user_id][chat_id].append({"role": "assistant", "content": assistant_response})

        _save_chat_to_disk()


def get_chat_history(user_id: str, chat_id: str):
    with _chat_lock:
        history = CHAT_HISTORY_STORE.get(user_id, {}).get(chat_id, [])
        return list(history)

def get_all_user_ids():
    with _chat_lock:
        return list(CHAT_HISTORY_STORE.keys())