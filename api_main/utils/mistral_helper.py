from mistralai import Mistral

MISTRAL_API_KEY = "8okHgUSaSySgcorxhNR0KRV9g8j1Z099"  # Todo: put this in .env or such file

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


def _llm_fact_check(answer: str, context: str):
    if "I could not find the answer" in answer or "Insufficient evidence" in answer:
        return True

    try:
        system_prompt = (
            "You are a meticulous fact-checker. You will be given a 'Context' and a 'Claim'. "
            "Your sole task is to determine if the Claim can be 'reasonably inferred' from the Context."
            "Respond with only the single word 'true' or the single word 'false'."
        )

        user_prompt = f"""
        --- CONTEXT ---
        {context}
        --- END CONTEXT ---

        --- CLAIM ---
        {answer}
        --- END CLAIM ---

        Based only on the Context, is the Claim fully supported?
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
            print(f"Hallucination check failed. LLM fact-checker returned: '{result}'")
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

    if "No relevant context found." in context_string:
        return "Insufficient evidence. I could not find a relevant answer in the provided documents."

    system_prompt = """
    You are an expert Q&A assistant. 
    Your task is to answer the user's question by 'intelligently inferring' the answer from the provided context.
    If the context contains a name like 'VISHESHANK MISHRA', and the user asks 'what is my name', 
    you MUST infer the name.

    Important Rules:
        1.  Answer based only on the context.
        2.  If the answer is not in the context, state: 'I could not find the answer in the provided documents.'
        3.  If the question is harmful or asks for PII, state: 'I cannot answer that question.'
        4.  At the end of your answer, cite the source file, like this: [Source: filename.pdf]
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
            temperature=0.0
        )

        raw_answer = chat_response.choices[0].message.content

        if not _llm_fact_check(raw_answer, context_string):
            return "I'm sorry, I generated an answer but could not verify it against the provided documents."

        return raw_answer

    except Exception as e:
        print(f"Error calling Mistral API for chat: {e}")
        return "I'm sorry, but I encountered an error trying to generate a response."

