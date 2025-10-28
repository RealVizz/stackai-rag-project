from mistralai import Mistral

MISTRAL_API_KEY = "8okHgUSaSySgcorxhNR0KRV9g8j1Z099"  # Todo: put this in .env or such file

client = Mistral(api_key=MISTRAL_API_KEY)

def get_embeddings_from_str_list(text_chunks_list):
    if not text_chunks_list:
        return []

    try:
        response = client.embeddings.create(
            model="mistral-embed",  # todo: try out diff models, of diff size.
            inputs=text_chunks_list
        )

        embeddings = [data.embedding for data in response.data] # Extracting just embedding vectors from the response.
        return embeddings

    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        return []
