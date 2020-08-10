import copy
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

import config

def mean_target_encoding(data):

    # make a copy of the dataframe
    df = copy.deepcopy(data)

    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain"
        "capital.loss"
        "hours.per.week"
    ]

    # map targets to 0s and 1s
    target_mapping = {
        " <=50K" : 0,
        " >50K" : 1
    }
    df.loc[:, "Income"] = df.Income.map(target_mapping)

    features = [
        f for f in df.columns if f not in ("kfold", "Income")
    ]

    # fill all NaN values with NONE
    for col in features:
        # do not encode numrical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # now its time to label encode the features
    for col in features:
        if col not in num_cols:
            # init LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()

            # fit label encoder on all data
            lbl.fit(df[col])

            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes
    encoded_dfs = []

    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for column in features:
            # create dict of category: mean target
            mapping_dict = dict(
                df_train.groupby(column)["Income"].mean()
            )
            # column_enc is the new column we have with mean encoding
            df_valid.loc[
                :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [
        f for f in df.columns if f not in ("kfold", "Income")
    ]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )

    model.fit(x_train, df_train.Income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.Income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    # read data
    df = pd.read_csv(config.CEN_TRAINING_FILE_FOLDS)
    df = mean_target_encoding(df)

    for fold_ in range(5):
        run(df, fold_)