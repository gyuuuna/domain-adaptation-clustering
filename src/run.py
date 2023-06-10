from datetime import datetime
import pandas as pd
from src.clustering import kmeans as get_clusters
from src.model import FeedForwardNetwork
import torch

DESC = 'pretrained_1'
TRAIN_FILENAME = f'data/train_{DESC}'
TEST_FILENAME = f'data/test_{DESC}'

def process_data(data, embedding_type):
    print('processing data ...')
    sentences = data['sentence']
    
    labels_real = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
    labels_real = data[labels_real]
    labels_real = labels_real.idxmax(axis=1)
    
    embeddings = data[embedding_type]
    embeddings = [eval(embedding) for embedding in embeddings]
    
    ids = data['id']
    return sentences, labels_real, embeddings, ids

def normalize_embedding(embedding):
    embedding_norm = torch.norm(embedding, p=2)
    normalized_embedding = embedding / embedding_norm
    return normalized_embedding

def process_embeddings(embeddings, weights_filename):
    print('processing embeddings with ffn layer ...')
    ffn = FeedForwardNetwork(512, 512)
    ffn_weights = torch.load(weights_filename)
    ffn.load_state_dict(ffn_weights)
    embeddings = [normalize_embedding(ffn(torch.Tensor(embedding))).detach().numpy() for embedding in embeddings]
    return embeddings


def save_clusters(num_clusters, filename, embedding_type, weights_filename):
    
    print(f'retrieving {filename} {embedding_type} ...')
    data = pd.read_csv(f'{filename}.csv')[:10000]
    sentences, labels_real, embeddings, ids = process_data(data, embedding_type)
    embeddings = process_embeddings(embeddings, weights_filename)
    
    print(f'clustering ...')
    labels = get_clusters(embeddings, num_clusters)

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    result_file = open(f'result/cluster {time_string}.txt', mode='w', encoding='utf-8')
    print(f'processed {filename}, num_clusters={num_clusters}, embedding={embedding_type}', file=result_file)

    for text, cluster_label, label_real, id in zip(sentences, labels, labels_real, ids):
        print(f'[pred: {cluster_label}] [real: {label_real}] ({id}) {text}', file=result_file)
        
def main():
    save_clusters(4, TEST_FILENAME, 'USE', 'triplet_ffn_weights.pth')
    
if __name__ == "__main__":
    main()
    