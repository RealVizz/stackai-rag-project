import shutil
import uuid
from typing import List

from fastapi import UploadFile

from api_main.utils.chat_memory_helper import get_chat_history, add_chat_turn
from api_main.utils.keyword_db_helper import add_chunks_to_kw_db_index, search_keywords
from api_main.utils.mistral_helper import (get_embeddings_from_str_list, get_embedding_from_str, get_llm_response,
                                           merge_and_rerank)
from api_main.utils.pdf_helper import process_pdf, PDFProcessingError
from api_main.utils.vector_db_helper import add_embeddings, get_vector_store_chat_data
from api_main.utils.vector_db_helper import get_top_k_vector_results
from config.config import BASE_STORAGE_RAW_DATA_FOLDER


def _validate_pdf_type(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise PDFProcessingError(f"Unsupported file type: '{file.filename}'. Only '.pdf' files are accepted.")


def _generate_unique_filename_and_path(file: UploadFile, storage_path):
    file_uuid = f"{str(uuid.uuid4())}_{file.filename}"
    full_file_path = storage_path / file_uuid
    return file_uuid, full_file_path


def process_pdf_upload(user_id: str, chat_id: str, files: List[UploadFile]):
    results = []
    storage_path = BASE_STORAGE_RAW_DATA_FOLDER / f"user_id_{user_id}" / f"chat_id_{chat_id}"
    storage_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        full_file_path = None
        try:
            _validate_pdf_type(file)
            file_uuid, full_file_path = _generate_unique_filename_and_path(file, storage_path)

            with full_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            text_chunks = process_pdf(str(full_file_path))
            if not text_chunks:
                raise PDFProcessingError(
                    "No text could be processed from the PDF. The file might be empty or unreadable.")

            embeddings = get_embeddings_from_str_list(text_chunks)
            if not embeddings:
                raise ValueError("Failed to generate embeddings for the document.")

            new_chunk_data = add_embeddings(
                user_id=user_id, chat_id=chat_id, source_file_uuid=file_uuid, source_file_name=file.filename,
                chunks=text_chunks, embeddings=embeddings
            )

            add_chunks_to_kw_db_index(user_id, chat_id, new_chunk_data)

            results.append({
                "filename": file.filename,
                "status": "success",
                "message": "File uploaded successfully."
            })

        except (PDFProcessingError, ValueError) as e:
            if full_file_path and full_file_path.exists():
                full_file_path.unlink()

            print(f"Error processing file {file.filename} for user {user_id}: {e}")

            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            continue  # To move on to the next file.

        except Exception as e:
            if full_file_path and full_file_path.exists():
                full_file_path.unlink()

            print(f"Unexpected error with file {file.filename} for user {user_id}: {e}")

            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"An unexpected service error occurred: {e}"
            })
            continue

    return results


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
    reranked_results = merge_and_rerank(vector_res, keyword_res)
    current_chat_history = get_chat_history(user_id, chat_id)

    resp = get_llm_response(
        user_query=query_str,
        reranked_results=reranked_results,
        chat_history=current_chat_history,
        max_tokens=8192
    )

    add_chat_turn(user_id, chat_id, query_str, resp)

    return resp
