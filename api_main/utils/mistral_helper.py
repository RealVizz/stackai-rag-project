import re

from mistralai import Mistral

from config.config import MISTRAL_API_KEY

client = Mistral(api_key=MISTRAL_API_KEY)


def _get_embeddings(inputs: list[str]) -> list[list[float]]:
    if not inputs:
        return []
    try:
        response = client.embeddings.create(model="mistral-embed", inputs=inputs)
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error calling Mistral API: {e}")
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


def _llm_fact_check(answer, context, chat_history):
    if "I could not find the answer" in answer or "Insufficient evidence" in answer or "I cannot answer" in answer:
        return True

    claim_to_check = re.sub(r"\[Source[^]]+]", "", answer).strip()
    if not claim_to_check:
        return True

    try:
        system_prompt = (
            "You are a meticulous fact-checker. "
            "You will be given 'Document Context', 'Chat History', and a 'Claim'. "
            "Your sole task is to determine if the Claim can be 'reasonably inferred', "
            "from EITHER the Document Context OR the Chat History. "
            "Respond with 'only' the single word 'true' or the single word 'false'."
        )

        history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
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
        {claim_to_check}
        --- END CLAIM ---

        Based 'only' on the Document Context OR the Chat History provided, 
        is the Claim reasonably supported or inferrable?
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.0
        )

        result = response.choices[0].message.content.strip().lower()

        if result == "true":
            return True
        else:
            print(f"Hallucination check failed. LLM fact-checker returned: '{result}' for claim '{claim_to_check}'")
            return False

    except Exception as e:
        print(f"Error during LLM fact-check: {e}")
        return False


def _manage_token_limit(messages: list[dict], max_tokens: int):
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


def get_llm_response(
        user_query: str,
        vector_results: list[dict],
        keyword_results: list[dict],
        chat_history: list[dict] = None,
        max_tokens: int = 8192
):
    if chat_history is None:
        chat_history = []

    context_string = _build_context_string(vector_results, keyword_results)

    system_prompt = """
    You are an expert Q&A assistant.
    Your task is to answer the user's question using the provided context and chat history.

    Rules:
    1.  Base your answer 'only' on the context or chat history. Prioritize context if available.
    2.  If the answer cannot be found, state: 'I could not find the answer in the provided documents or chat history.'
    3.  If the question is harmful or asks for PII, state: 'I cannot answer that question.'
    4.  If answering from context, cite the source file like this: [Source: filename.pdf]
    5.  If answering only from chat history, do NOT add a source citation.
    """

    user_prompt = _build_user_prompt(user_query, context_string)

    messages = list()
    messages.append({"role": "system", "content": system_prompt})
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    messages_to_send = _manage_token_limit(messages, max_tokens)

    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages_to_send,
            temperature=0.1
        )

        raw_answer = chat_response.choices[0].message.content

        if not _llm_fact_check(raw_answer, context_string, chat_history):
            if "No relevant context found." in context_string and not chat_history:
                 return ("Insufficient evidence. "
                         "I could not find a relevant answer in the provided documents or chat history.")

            return "I'm sorry, I generated an answer but could not verify it against the provided information."

        return raw_answer

    except Exception as e:
        print(f"Error calling Mistral API for chat: {e}")
        return "I'm sorry, but I encountered an error trying to generate a response."
