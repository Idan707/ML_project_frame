import io
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from yaml import tokens

import nlp_config


def load_vectors(fname):
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    """
    Given a sentence and other information,
    this function returns embedding for the whole sentence
    :param s: sentence, string
    :param embedding_dict: dictionary word:vector
    :param stop_words: list of stop words, if any
    :param tokenizer: a tokenization function
    """
    # convert sentence to string and lowercase it
    words = str(s).lower()

    # tokenize the sentence
    words = tokenizer(words)

    # remove stop words tokens
    words = [w for w in words if not w in stop_words]

    # keep only alpha-numeric tokens
    words = [w for w in words if w.isalpha()]

    # init empty list to store embeddings
    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])

    # if we dont have any vectors, return zeros
    if len(M) == 0:
        return np.zeros(300)

    # convert list of embeddings to array
    M = np.array(M)

    # calculate sum over axis=0
    v = M.sum(axis=0)

    # return normalized vector
    return v / np.sqrt((v ** 2).sum())

if __name__ == "__main__":
    df = pd.read_csv(nlp_config.TRAINING_FILE)
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # randomize the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # load embedding into memory
    print("Loading embeddings")
    embeddings = load_vectors("./input/crawl-300d-2M.vec")

    # create sentence embeddings
    print("Creating sentence vectors")
    vectors = []
    for review in df.review.values:
        vectors.append(
            sentence_to_vec(
                s = review,
                embedding_dict = embeddings,
                stop_words = [],
                tokenizer = word_tokenize
            )
        )
    
    vectors = np.array(vectors)

    # fetch labels
    y = df.sentiment.values

    # init the kfold class 
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training fold: {fold_}")
        # temp dfs for train and test
        xtrain = vectors[t_, :]
        ytrain = y[t_]

        xtest = vectors[v_, :]
        ytest = y[v_]

        # init model
        model = linear_model.LogisticRegression()

        # fit model
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(ytest, preds)
        print(f"Accuracy = {accuracy}")
        print("")