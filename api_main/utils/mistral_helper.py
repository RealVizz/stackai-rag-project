import re

from mistralai import Mistral

from api_main.utils.prompts import MAIN_SYSTEM_PROMPT, FACT_CHECK_SYSTEM_PROMPT
from config.config import MISTRAL_API_KEY

client = Mistral(api_key=MISTRAL_API_KEY)


def _get_embeddings(inputs: list[str]) -> list[list[float]]:
    if not inputs:
        return []
    try:
        response = client.embeddings.create(model="mistral-embed", inputs=inputs)
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error calling Mistral API for embeddings: {e}")
        return []


def get_embeddings_from_str_list(text_chunks_list: list[str]) -> list[list[float]]:
    return _get_embeddings(text_chunks_list)


def get_embedding_from_str(query_text: str) -> list[float]:
    embeddings = _get_embeddings([query_text])
    return embeddings[0] if embeddings else []


def _build_context_string(vector_results: list[dict], keyword_results: list[dict]):
    all_chunks = {}

    for item in vector_results + keyword_results:
        text = item.get("text_chunk")
        if text:
            source = item.get("source_file_name", "unknown")
            all_chunks[text] = source

    if not all_chunks:
        return "No relevant context found."

    context_with_citations = []
    for i, (text, source) in enumerate(all_chunks.items()):
        context_with_citations.append(f"[Source {i + 1}, file: {source}]:\n{text}")

    return "\n\n---\n\n".join(context_with_citations)


def _build_user_prompt(user_query: str, context_string: str):
    return f"""
    --- CONTEXT ---
    {context_string}
    --- END CONTEXT ---

    User Question: {user_query}
    """


def _manage_token_limit(messages: list[dict], max_tokens: int):
    # Estimating: 1 token ~= 3 chars, keeping it very conservative (for now).
    estimated_length = sum(len(msg["content"]) for msg in messages)

    if estimated_length < max_tokens * 3:
        return messages

    system_prompt = messages[0]
    current_user_prompt = messages[-1]
    history = messages[1:-1]

    while estimated_length > max_tokens * 3 and len(history) > 0:
        removed_message = history.pop(0)
        estimated_length -= len(removed_message["content"])

    return [system_prompt] + history + [current_user_prompt]


def _execute_llm_call(messages: list[dict], temperature: float = 0.1) -> str:
    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=temperature
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Mistral chat API: {e}")
        raise


def _prepare_fact_check_messages(context: str, history: list[dict], claim: str) -> list[dict]:
    history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    if not history_string:
        history_string = "No chat history provided."

    user_prompt = f"""
        --- DOCUMENT CONTEXT ---
        {context}
        --- END DOCUMENT CONTEXT ---

        --- CHAT HISTORY ---
        {history_string}
        --- END CHAT HISTORY ---

        --- CLAIM ---
        {claim}
        --- END CLAIM ---

        Based 'only' on the Document Context OR the Chat History provided, 
        is the Claim reasonably supported or inferrable?
        """

    return [
        {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def _llm_fact_check(answer: str, context: str, chat_history: list[dict]) -> bool:
    if "I could not find the answer" in answer or "Insufficient evidence" in answer or "I cannot answer" in answer:
        return True

    claim_to_check = re.sub(r"\[Source[^]]+]", "", answer).strip()  # Removing citations from the claim.

    if not claim_to_check:  # If there's no claim (ex: empty string), considering it to be valid.
        return True

    try:
        messages = _prepare_fact_check_messages(context, chat_history, claim_to_check)
        result = _execute_llm_call(messages, temperature=0.0).strip().lower()
        if result == "true":
            return True
        else:
            print(f"Hallucination check failed. LLM fact-checker returned: '{result}' for claim '{claim_to_check}'")
            return False

    except Exception as e:
        print(f"Error during LLM fact-check call: {e}")
        return False


def _prepare_main_llm_messages(user_query: str, context_string: str, chat_history: list[dict], max_tokens: int):
    user_prompt = _build_user_prompt(user_query, context_string)

    messages = list()
    messages.append({"role": "system", "content": MAIN_SYSTEM_PROMPT})  # Use imported prompt
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    return _manage_token_limit(messages, max_tokens)


def get_llm_response(user_query: str, vector_results: list[dict], keyword_results: list[dict],
                     chat_history: list[dict] = None, max_tokens: int = 8192):
    if chat_history is None:
        chat_history = []

    try:
        context_string = _build_context_string(vector_results, keyword_results)
        messages_to_send = _prepare_main_llm_messages(user_query, context_string, chat_history, max_tokens)
        raw_answer = _execute_llm_call(messages_to_send, temperature=0.1)

        if not _llm_fact_check(raw_answer, context_string, chat_history):
            if "No relevant context found." in context_string and not chat_history:
                return ("Insufficient evidence. "
                        "I could not find a relevant answer in the provided documents or chat history.")

            return "I'm sorry, I generated an answer but could not verify it against the provided information."

        return raw_answer

    except Exception as e:
        print(f"Error in get_llm_response: {e}")
        return "I'm sorry, but I encountered an error trying to generate a response."
