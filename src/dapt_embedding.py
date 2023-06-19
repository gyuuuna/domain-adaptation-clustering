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
from keras.utils import pad_sequences
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from keras.models import Model
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
        
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss

def build_network():
    
    network = Sequential()
    network.add(Embedding(name="synopsis_embedd",input_dim =len(t.word_index)+1, 
                        output_dim=len(embeddings_index['no']),weights=[embedding_matrix], 
                        input_length=train_q1_seq.shape[1],trainable=False))
    network.add(LSTM(64,return_sequences=True, activation="relu"))
    network.add(Flatten())
    network.add(Dense(128, activation='relu',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer='he_uniform'))
    
    network.add(Dense(2, activation=None,
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

    return network

def run():
    task_initial = pd.read_csv('dev.csv')['text'].values.tolist()
    non_task_initial = pd.read_csv('cnn_full.csv')['text'].values.tolist()

    task = task_initial
    n = int(len(task) / 3)
    task_one = task[:n]
    task_two = task[n:2*n]
    task_three = task[2*n:]
    non_task_one = non_task_initial[:n]
    non_task_two = non_task_initial[n:2*n]
    non_task_three = non_task_initial[2*n:len(task)]

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
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    X_train['text'] = X_train[['text1', 'text2']].apply(lambda x: str(x[0])+" "+str(x[1]), axis=1)


    # Glove

    t = Tokenizer()
    t.fit_on_texts(X_train['text'].values)

    X_train['text1'] = X_train['text1'].astype(str)
    X_train['text2'] = X_train['text2'].astype(str)
    X_val['text1'] = X_val['text1'].astype(str)
    X_val['text2'] = X_val['text2'].astype(str)

    train_q1_seq = t.texts_to_sequences(X_train['text1'].values)
    train_q2_seq = t.texts_to_sequences(X_train['text2'].values)
    val_q1_seq = t.texts_to_sequences(X_val['text1'].values)
    val_q2_seq = t.texts_to_sequences(X_val['text2'].values)

    max_len = 200
    train_q1_seq = pad_sequences(train_q1_seq, maxlen=max_len, padding='post')
    train_q2_seq = pad_sequences(train_q2_seq, maxlen=max_len, padding='post')
    val_q1_seq = pad_sequences(val_q1_seq, maxlen=max_len, padding='post')
    val_q2_seq = pad_sequences(val_q2_seq, maxlen=max_len, padding='post')

    embeddings_index = {}
    with open('glove.6B.300d.txt') as f:
        for line in f:
            values = line.split()
            if len(values) != 301:  # Adjust the value based on the dimension of your word embeddings
                continue
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue
    print('Found %s word vectors.' % len(embeddings_index))


    not_present_list = []
    vocab_size = len(t.word_index) + 1
    print('Loaded %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, len(embeddings_index['no'])))
    for word, i in notebook.tqdm(t.word_index.items()):
        if word in embeddings_index.keys():
            embedding_vector = embeddings_index.get(word)
        else:
            not_present_list.append(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.zeros(300)
        
    input_1 = Input(shape=(train_q1_seq.shape[1],))
    input_2 = Input(shape=(train_q2_seq.shape[1],))

    network = build_network()

    encoded_input_1 = network(input_1)
    encoded_input_2 = network(input_2)

    distance = Lambda(euclidean_distance)([encoded_input_1, encoded_input_2])

    # Connect the inputs with the outputs
    model = Model([input_1, input_2], distance)

    model.compile(loss=contrastive_loss, optimizer=Adam(0.001))

    y_train = np.asarray(y_train).astype('float32')
    y_val = np.asarray(y_val).astype('float32')

    model.fit([train_q1_seq,train_q2_seq],y_train.reshape(-1,1), epochs = 10, 
            batch_size=64,validation_data=([val_q1_seq, val_q2_seq],y_val.reshape(-1,1)))

    # Save model for further use
    # serialize model to JSON
    model_json = model.to_json()
    with open("siamesemodel-contrastive-loss-dev.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("siamesemodel-contrastive-loss-dev.h5")
    print("Saved model to disk")

    # load json and create model
    # json_file = open('siamesemodel-contrastive-loss.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("siamesemodel-contrastive-loss.h5")
    # print("Loaded model from disk")
    
if __name__=='__main__':
    run()