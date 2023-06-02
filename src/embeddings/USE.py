import tensorflow_hub as hub
# Load pre-trained universal sentence encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_embeddings(sentences):
    embeddings = embed(sentences)
    return list(list(embedding) for embedding in embeddings)