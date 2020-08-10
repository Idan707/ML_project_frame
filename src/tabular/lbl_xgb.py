import pandas as pd 
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

import config

def run(fold):
    # load the full trianing data with folds
    df = pd.read_csv(config.CEN_TRAINING_FILE_FOLDS)

    # list of numrical columns
    num_cols = [
        "fnlwgt",
        "Age",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    # drop numrical columns
    df = df.drop(num_cols, axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }
    df.loc[:, "Income"] = df.Income.map(target_mapping)

    # all columns are features except kfold & income columns
    features = [
        f for f in df.columns if f not in ("kfold", "Income")
    ]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # now its time to lavel encode the features
    for col in features:

        # init LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # fit label encoder on all data
        lbl.fit(df[col])

        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # init xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )

    # fit model on training data (ohe)
    model.fit(x_train, df_train.Income.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.Income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

