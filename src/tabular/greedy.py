import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

import config


class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    We will need to adjust to different data sets.
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns AUC.
        We fit the data and calculate AUC on same data, means we overfitting here.
        But this is also the way to achieve greedy selection.

        If we want to implement it in really correct way,
        Calculate OOF AUC and return mean AUC over K fold.

        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """

        # fit model and calculate auc on the same data
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        """
        This function dose the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        # init good feature list
        # and best scores to keep track of both
        good_features = []
        best_scores = []

        # calculate the number of features
        num_features = X.shape[1]

        # infinite loop
        while True:
            # init best feature and score of this loop
            this_feature = None
            best_score = 0

            # loop over all features
            for feature in range(num_features):
                # if feature in good_features list, skip
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)
                # if score is greater than the best score of this loop
                # change best score and best feature
                if score > best_score:
                    this_feature = feature
                    best_score = score

            if this_feature != None:
                good_features.append(this_feature)
                best_score.append(best_score)

            if len(best_score) > 2:
                if best_score[-1] < best_score[-2]:
                    break

        return best_score[:-1], good_features[:-1]

    def __call__(self, X, y):
        scores, features = self._feature_selection(X, y)

        return X[:, features], scores

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=50)
    X_transformed, scores = GreedyFeatureSelection()(X, y)