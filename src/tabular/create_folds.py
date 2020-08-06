import pandas as pd
import numpy as np
import os
from sklearn import model_selection

import config

if __name__ == "__main__":

    # read training data
    df = pd.read_csv(config.TRAINING_FILE)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.label.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for feature, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = feature

    # save the new csv with kfold column
    HOME_DIR = os.path.expanduser("~")
    #SAVE_TO = os.path.join(HOME_DIR, "git", "ML_project_frame", "input", "mnist_train_folds.csv")
    SAVE_TO = os.path.join(HOME_DIR, "Documents", "GitHub", "ML_project_frame", "input", "mnist_train_folds.csv")
    df.to_csv(SAVE_TO, index=False)