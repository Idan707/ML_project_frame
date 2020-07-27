import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils


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

        inp = layers