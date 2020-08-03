import io
import torch

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn import metrics
from nlp.fasttext import embeddings

import nlp_config
import dataset
import engine
import lstm

def load_vectors(fname):
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split)
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dict with word:index_value
    :param embedding_dict: a dict with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    # init matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embedding 
        # update the matrix, if no, the vector is zeros
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]

    return embedding_matrix

def run(df, fold):
    """
    Run training and validation for a given fold
    and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    """

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df  = df[df.kfold == fold].reset_index(drop=True)
    print("Fitting tokenizer")

    # we use tf.keras for tokenization
    # you can use your own and get rid of tensorflow

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    #TODO:continue with code here

