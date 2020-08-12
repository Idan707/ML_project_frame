import numpy as np
import stats 

from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

def mean_predictions(probas):
    """
    Create mean predictions
    :param probas: 2-d array of probability values
    :return: mean probability
    """
    return np.mean(probas, axis=1)

def max_voting(preds):
    """
    Creat max prediction
    :param pred: 2-d array of prediction values
    :return: max voted predictions
    """
    idx = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)

def rank_mean(probas):
    """
    Create mean predictions using ranks
    :param probas: 2-d array of probability values
    :return: mean ranks
    """

    ranked = []
    for i in enumerate(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)
        
    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)

class OptimizeAUC:
    """
    Class for optimizing AUC.
    This class is all you need tp find the best weights for
    any model and for any metric anf for any type of predictions.
    with very small changes, this class can be used for optimiztion of
    weights in ensemble models of _any_ type of predictions
    """
    def __init__(self):
        self.conf_ = 0

    def _auc(self, coef, X, y):
        """
        This functions calculates and returns AUC.
        :param coef: coef list, of the same length as number of models
        :param X: predictions, in this case a 2d array
        :param y: targets, in out case binary 1d array
        """
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)

        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions