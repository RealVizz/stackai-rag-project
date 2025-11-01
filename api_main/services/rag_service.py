from pathlib import Path

from fastapi import UploadFile

from api_main.utils.chat_memory_helper import get_chat_history, add_chat_turn
from api_main.utils.keyword_db_helper import add_chunks_to_index, search_keywords
from api_main.utils.mistral_helper import get_embeddings_from_str_list, get_embedding_from_str, get_llm_response
from api_main.utils.pdf_helper import process_pdf, PDFProcessingError
from api_main.utils.vector_db_helper import add_embeddings, get_vector_store_chat_data
from api_main.utils.vector_db_helper import get_top_k_vector_results


def process_pdf_upload(user_id: str, chat_id: str, file: UploadFile, file_uuid: str, full_file_path: Path) -> str:
    try:
        text_chunks = process_pdf(str(full_file_path))
        if not text_chunks:
            raise PDFProcessingError("No text could be processed from the PDF. The file might be empty or unreadable.")

        embeddings = get_embeddings_from_str_list(text_chunks)

        if not embeddings:
            # This is a server-side issue, not a user file issue
            raise ValueError("Failed to generate embeddings for the document.")

        new_chunk_data = add_embeddings(
            user_id=user_id,
            chat_id=chat_id,
            source_file_uuid=file_uuid,
            source_file_name=file.filename,
            chunks=text_chunks,
            embeddings=embeddings
        )

        add_chunks_to_index(user_id, chat_id, new_chunk_data)

        return "File uploaded successfully."

    except (PDFProcessingError, ValueError) as e:
        if full_file_path and full_file_path.exists():
            full_file_path.unlink()
        print(f"Error in processing PDF for user {user_id}, chat {chat_id}: {e}")
        raise e  # Re-raise the original exception

    except Exception as e:
        if full_file_path and full_file_path.exists():
            full_file_path.unlink()
        print(f"Unexpected error in service layer for user {user_id}: {e}")
        raise ValueError(f"An unexpected service error occurred: {e}")


def process_query(user_id: str, chat_id: str, query_str: str) -> str:
    query_embeddings = get_embedding_from_str(query_str)
    if not query_embeddings:
        raise ValueError("Failed to generate query embedding.")

    all_chat_chunks = get_vector_store_chat_data(user_id, chat_id)

    vector_res = get_top_k_vector_results(
        chat_chunks_data=all_chat_chunks,
        query_embedding=query_embeddings,
        k=5,
        threshold=0.5
    )

    keyword_res = search_keywords(user_id, chat_id, query_str, all_chat_chunks)
    current_chat_history = get_chat_history(user_id, chat_id)

    resp = get_llm_response(
        user_query=query_str,
        vector_results=vector_res,
        keyword_results=keyword_res,
        chat_history=current_chat_history,
        max_tokens=8192
    )

    add_chat_turn(user_id, chat_id, query_str, resp)

    return resp
