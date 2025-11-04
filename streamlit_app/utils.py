from typing import Any

import requests

# Configuration for the FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:11111"


def get_all_chat_ids_for_user(user_id: str) -> list[str]:
    """Fetches all chat IDs for a given user from the backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/chats/", json={"user_id": user_id})
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get("chat_ids", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching chat IDs for user {user_id}: {e}")
        return []


def get_chat_history(user_id: str, chat_id: str) -> list[dict[str, str]]:
    """Fetches the chat history for a specific user and chat ID from the backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/get-chat-history/", json={
            "user_id": user_id,
            "chat_id": chat_id
        })
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get("history", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching chat history for user {user_id}, chat {chat_id}: {e}")
        return []


def query_backend(user_id: str, chat_id: str, query_str: str) -> str:
    """Sends a query to the RAG backend and returns the response."""
    try:
        response = requests.post(f"{BACKEND_URL}/query/", json={
            "user_id": user_id,
            "chat_id": chat_id,
            "query_str": query_str
        })
        response.raise_for_status()
        return response.json().get("resp", "Error: No response from LLM.")
    except requests.exceptions.RequestException as e:
        print(f"Error querying backend for user {user_id}, chat {chat_id}: {e}")
        return f"Error: Could not connect to backend or invalid response: {e}"


def upload_pdf_to_backend(user_id: str, chat_id: str, files: list[Any]) -> list[dict[str, str]]:
    """Uploads PDF files to the backend for processing."""
    if not user_id or not chat_id:
        return [{"filename": "N/A", "status": "error", "message": "User ID and Chat ID are required for upload."}]
    uploaded_files_data = []
    for file_obj in files:
        uploaded_files_data.append(("files", (file_obj.name, file_obj.getvalue(), file_obj.type)))

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload-pdf/",
            files=uploaded_files_data,
            data={
                "user_id": user_id,
                "chat_id": chat_id
            }
        )
        response.raise_for_status()
        return response.json().get("message", [])
    except requests.exceptions.RequestException as e:
        print(f"Error uploading PDF for user {user_id}, chat {chat_id}: {e}")
        return [{"filename": "N/A", "status": "error", "message": f"Upload failed: {e}"}]


def get_all_user_ids() -> list[str]:
    """Fetches all user IDs from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/users/")
        response.raise_for_status()
        return response.json().get("user_ids", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching all user IDs: {e}")
        return []
