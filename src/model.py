import pandas as pd
import random
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.data import RandomPairDataset, TripletDataset

class FeedForwardNetwork(nn.Module):
    '''
    Temporary
    '''
    def __init__(self, input_size, hidden_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

class TripletDistanceMetric():
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    def __init__(self, input_size, hidden_size, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 5):
        super(TripletLoss, self).__init__()
        self.ffn = FeedForwardNetwork(input_size, hidden_size)
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin
      
    def get_ffn_weights(self):
        return self.ffn.state_dict()

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break
        return {'distance_metric': distance_metric_name, 'triplet_margin': self.triplet_margin}

    def forward(self, vectors):
        anchor, pos, neg = vectors
        rep_anchor, rep_pos, rep_neg = self.ffn(anchor), self.ffn(pos), self.ffn(neg)
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()

def get_triple_dataset(dataframe):
    vectors = dataframe['USE']
    labels = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
    labels = dataframe[labels]
    labels = labels.idxmax(axis=1)
    
    tripletDataset = TripletDataset(vectors, labels)
    return tripletDataset

def train(dataset, model, optimizer, dataset_size=10000):
    num_epochs = 30
    batch_size = 20
    data_loader = DataLoader(Subset(dataset, range(dataset_size)), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch in data_loader:
            anchor, positive, negative = batch
            optimizer.zero_grad()
            loss = model([anchor, positive, negative])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
def main():
    df_train = pd.read_csv('data/research-dataset-pretrained-weights/train_pretrained_1.csv')
    df_test = pd.read_csv('data/research-dataset-pretrained-weights/test_pretrained_1.csv')
    tripletDataset = get_triple_dataset(df_train)
    
    input_size = 512
    hidden_size = 512
    
    model = TripletLoss(input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train(tripletDataset, model, optimizer, dataset_dize=10000)
    
    ffn_weights = model.get_ffn_weights()
    torch.save(ffn_weights, 'ffn_weights.pth')

    ffn_model = FeedForwardNetwork(input_size, hidden_size)
    ffn_weights = torch.load('ffn_weights.pth')
    ffn_model.load_state_dict(ffn_weights)
    
    '''
    Require a Test Code
    '''

    score1 = 0
    score2 = 0

    test_dataset = RandomPairDataset(df_test)
    start = 0
    end = len(test_dataset)
    size = end-start

    def normalize_embedding(embedding):
        embedding_norm = torch.norm(embedding, p=2)
        normalized_embedding = embedding / embedding_norm
        return normalized_embedding

    for pair, target in Subset(test_dataset, range(start, end)):
        item1 = pair[0]
        item2 = pair[1]
        d1 = TripletDistanceMetric.EUCLIDEAN(item1.view(1, -1), item2.view(1, -1))[0].float()
        output1 = normalize_embedding(ffn_model(item1))
        output2 = normalize_embedding(ffn_model(item2))
        d2 = TripletDistanceMetric.EUCLIDEAN(output1.view(1, -1), output2.view(1, -1))[0].float()

        if d1<1 and target or d1>=1 and not target:
            score1+=1
        if d2<1 and target or d2>=1 and not target:
            score2+=1

    print(score1/size)
    print(score2/size)
    
if __name__=='__main__':
    main()