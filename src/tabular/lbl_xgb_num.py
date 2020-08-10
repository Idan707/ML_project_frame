import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

import config

def run(fold):
    # load the full training data with folds
    df = pd.read_csv(config.CEN_TRAINING_FILE_FOLDS)

    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "Age",
        "capital-gain"
        "capital-loss"
        "hours-per-week"
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

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values

    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1
    )

    model.fit(x_train, df_train.Income.values)

    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.Income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)