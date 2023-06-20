from tqdm import trange, notebook
import pandas as pd
import numpy as np
import random
import warnings
import time
import datetime
import re
import string
import itertools
import pickle
import joblib
import nltk
import csv


import tensorflow as tf
import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from keras.models import Model
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

base_url = ''

task_initial = pd.read_csv(base_url+'train.csv')['text'].values.tolist()
non_task_initial = pd.read_csv(base_url+'cnn_full.csv')['text'].values.tolist()

task = min(task_initial, non_task_initial)
n = int(len(task) / 3)
task_one = task_initial[:n]
task_two = task_initial[n:2*n]
task_three = task_initial[2*n:3*n]
non_task_one = non_task_initial[:n]
non_task_two = non_task_initial[n:2*n]
non_task_three = non_task_initial[2*n:3*n]

# Creating pairs of data for siamese training => label 1 if pairs from same class otherwise 0
df2 = pd.DataFrame(columns=['text1', 'text2', 'label'])

for idx, data in notebook.tqdm(enumerate(task_one)):
    data1 = data
    data2 = task_two[idx]
    data3 = non_task_one[idx]
    df2.loc[len(df2)] = [data1, data2, 1]
    df2.loc[len(df2)] = [data1, data3, 0]

for idx, data in notebook.tqdm(enumerate(non_task_two)):
    data1 = data
    data2 = non_task_three[idx]
    data3 = task_three[idx]
    df2.loc[len(df2)] = [data1, data2, 1]
    df2.loc[len(df2)] = [data1, data3, 0]
    
X_train, X_val, y_train, y_val = train_test_split(df2[['text1', 'text2']], df2['label'], test_size=0.2, random_state=0)
X_train['text'] = X_train[['text1', 'text2']].apply(lambda x: str(x[0])+" "+str(x[1]), axis=1)

# load json and create model
json_file = open(base_url + 'siamesemodel-contrastive-loss.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
embedding_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
embedding_model.load_weights(base_url + 'siamesemodel-contrastive-loss.h5')
print("Loaded model from disk")

test_df = pd.read_csv(base_url+'AG_test.csv', header=None)
train_df = pd.read_csv(base_url+'AG_train.csv')
domain_df = pd.read_csv(base_url+'cnn_full.csv')

t = Tokenizer()
t.fit_on_texts(X_train['text'].values)

def text_to_vector(text):
    vector = t.texts_to_sequences([text])
    vector = pad_sequences(vector,maxlen=200)
    return vector

def get_distance(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    prediction = embedding_model.predict([vec1, vec2])
    return prediction[0][0].item()

def knn_selection(query_vector, data_vectors, k):
    distances = [get_distance(query_vector, data_vector) for data_vector in data_vectors]
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:k]
    return top_indices

train_1 = train_df[train_df['label'] == 1]['text']
train_2 = train_df[train_df['label'] == 2]['text']
train_3 = train_df[train_df['label'] == 3]['text']
train_4 = train_df[train_df['label'] == 4]['text']
domain_data = domain_df['text']

k = 1
num_samples_train = 5
num_samples_domain = 1000

for train_data in train_1.sample(num_samples_train):
    top_indices = knn_selection(train_data, domain_data.sample(n=num_samples_domain), k)
    top_texts = domain_data[top_indices]
    print(top_texts)
    for text in top_texts:
        new_record = {'text': text, 'label': 1}
        train_df = train_df.append(new_record, ignore_index=True)
        
for train_data in train_2.sample(num_samples_train):
    top_indices = knn_selection(train_data, domain_data.sample(n=num_samples_domain), k)
    top_texts = domain_data[top_indices]
    print(top_texts)
    for text in top_texts:
        new_record = {'text': text, 'label': 2}
        train_df = train_df.append(new_record, ignore_index=True)
        
for train_data in train_3.sample(num_samples_train):
    top_indices = knn_selection(train_data, domain_data.sample(n=num_samples_domain), k)
    top_texts = domain_data[top_indices]
    print(top_texts)
    for text in top_texts:
        new_record = {'text': text, 'label': 3}
        train_df = train_df.append(new_record, ignore_index=True)
        
for train_data in train_4.sample(num_samples_train):
    top_indices = knn_selection(train_data, domain_data.sample(n=num_samples_domain), k)
    top_texts = domain_data[top_indices]
    print(top_texts)
    for text in top_texts:
        new_record = {'text': text, 'label': 4}
        train_df = train_df.append(new_record, ignore_index=True)
        
