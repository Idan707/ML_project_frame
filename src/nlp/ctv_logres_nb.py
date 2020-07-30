import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch lables
    y = df.sentiment.values

    # init kfold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill kfold
    for col, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = col

    for fold_ in range(5):
        # temp df for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # init CountVectorizer with NLTK's word_tokenize
        count_vec  = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )

        # fit count_vec on training data reviews
        count_vec.fit(train_df.review)

        # transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        # init logistic regression model
        model = linear_model.LogisticRegression() #naive_bayes.MultinomialBN()
        model.fit(xtrain, train_df.sentiment)
        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")