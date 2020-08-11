import os
import gc
import joblib
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import scipy
from sklearn import metrics, preprocessing
from sklearn.utils import validation
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

import config


def create_model(data, catcols):
    """
    This function returns a complied tf.keras model
    for entity embeddings
    :param data: this is pandas dataframe
    :param catcols: list of categorical column names
    :return: compiled tf.keras model
    """
    inputs = []
    outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out  = layers.Embedding(num_unique_values + 1, embed_dim, name = c)(inp)

        # 1-d spatial dropout is the standard for emebedding layers
        # you can use it in NLP task too
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding
        # this becomes our output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)

        # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layers
    # from now on you can try different architechtures
    # if we have a numrical features, you shold add here
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax and treating it as a two class problem
    y = layers.Dense(2, activation="softmax")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    # complie the model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def run(fold):
    # load training data with folds
    df = pd.read_csv(config.CAT_TRAINING_FILE_FOLDS)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # encode all features with label encoder individually
    # in live setting you need to save all label encoders
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # creat tf.keras model
    model = create_model(df, features)

    # out features are lists of lists
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]

    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    # fetch target columns
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # convert target columns to categories
    # this is just binarization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(xtrain,
              ytrain_cat,
              validation_data=(xvalid, yvalid_cat),
              verbose=1,
              batch_size=1024,
              epochs=3
    )

    # genrate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]

    # print roc auc score
    print(metrics.roc_auc_score(yvalid, valid_preds))

    K.clear_session()

if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)