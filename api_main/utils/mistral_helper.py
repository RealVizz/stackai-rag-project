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
