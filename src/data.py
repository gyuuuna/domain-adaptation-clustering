import pandas as pd
import random
import torch
from torch.utils.data import Dataset

class RandomPairDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe['USE']
        # get labels of each records
        labels = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
        labels = dataframe[labels]
        self.labels = labels.idxmax(axis=1)
        self.pairs, self.targets = self.generate_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        target = self.targets[index]
        item1 = eval(self.dataframe.loc[pair[0]])
        item2 = eval(self.dataframe.loc[pair[1]])
        return torch.tensor([item1, item2]), torch.tensor(target)

    def generate_pairs(self):
        indices = self.dataframe.index.tolist()
        random.shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
        targets = [self.get_target_label(pair) for pair in pairs]
        return pairs, targets

    def get_target_label(self, pair):
        label1 = self.labels[pair[0]]
        label2 = self.labels[pair[1]]
        if label1 == label2:
            return -1
        else:
            return 1
        
class TripletDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

        self.anchor_indices, self.positive_indices, self.negative_indices = self._create_triplets()

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, index):
        anchor_index = self.anchor_indices[index]
        positive_index = self.positive_indices[index]
        negative_index = self.negative_indices[index]

        anchor = eval(self.vectors[anchor_index])
        positive = eval(self.vectors[positive_index])
        negative = eval(self.vectors[negative_index])

        return torch.Tensor(anchor), torch.Tensor(positive), torch.Tensor(negative)

    def _create_triplets(self):
        anchor_indices = []
        positive_indices = []
        negative_indices = []

        label_to_indices = {} 

        for index, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(index)

        for label in self.labels:
            anchor_index = self._get_random_sample(label_to_indices[label])
            positive_index = self._get_random_sample(label_to_indices[label])

            negative_label = self._get_random_label(label_to_indices, label)
            negative_index = self._get_random_sample(label_to_indices[negative_label])

            anchor_indices.append(anchor_index)
            positive_indices.append(positive_index)
            negative_indices.append(negative_index)

        return anchor_indices, positive_indices, negative_indices

    def _get_random_sample(self, indices):
        return indices[torch.randint(0, len(indices), (1,)).item()]

    def _get_random_label(self, label_to_indices, exclude_label):
        labels = list(label_to_indices.keys())
        labels.remove(exclude_label)
        return labels[torch.randint(0, len(labels), (1,)).item()]