from sentence_transformers import SentenceTransformer
# Load pre-trained sentence_bert model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(sentences):
    embeddings = model.encode(sentences)
    return embeddings
