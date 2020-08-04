import io
from nlp.ctv_logres_nb import accuracy
import torch

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn import metrics
from nlp.fasttext import embeddings, review

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

    # convert training data to sequences
    # for exmple: "bad movie" get converted to
    # [24, 27] where 24 is the index for bad and 27 is the index for movie
    xtrain = tokenizer.text_to_sequences(train_df.review.values)
    xtest = tokenizer.text_to_sequences(valid_df.review.values)

    # zero pad the training sequences given the maximum length
    # this padding is done on left hend side
    # if sequence is > MAX_LEN, it is truncated on left hend side too
    xtrain = tf.keras.preprocessing.pad_sequences(
        xtrain, maxlen=nlp_config.MAX_LEN
    )

    xtest = tf.keras.preprocessing.pad_sequences(
        xtest, maxlen=nlp_config.MAX_LEN
    )

    train_dataset = dataset.IMDBDataset(
        reviews=xtrain,
        targets=train_df.sentiment.values
    )

    # create torch dataloader for training 
    # torch dataloader loads the data using dataset
    # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=nlp_config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    valid_dataset = dataset.IMDBDataset(
        reviews=xtest,
        targets=valid_df.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=nlp_config.VALID_BATCH_SIZE,
        num_workers=1
    )

    print("Loading embeddings")
    embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(
        tokenizer.word_index, embedding_dict
    )

    # create torch device, since we use gpu, we are using cuda
    device = torch.device("cuda")

    # fetch our LSTM model
    model = lstm.LSTM(embedding_matrix)

    # send model to device
    model.to(device)

    # init Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Model")
    best_accuracy = 0
    early_stopping_counter = 0
    for epoch in range(nlp_config.EPOCHS):
        engine.train(train_data_loader, model, optimizer, device)
        outputs, targets = engine.evaluate(
                                valid_data_loader, model, device
        )

        # use threshold of 0.5
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
            f"FOLD{fold}, Epoch:{epoch}, Accuracy Score = {accuracy}"
        )

        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

        if early_stopping_counter > 2:
            break

if __name__ == "__main__":
    df = pd.read_csv(nlp_config.TRAINING_FILE)

    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)

