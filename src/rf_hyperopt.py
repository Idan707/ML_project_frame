import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

import config


def optimize(params, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs cv 
    and return negative accuracy score

    :param params: list of params from gp_minimize
    :param param_names: list of param names. order is important!
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)

    return -1 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)

    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # define a parameter space
    # now we use hyperopt
    param_space = {
        # quniform gives round(uniform(low, high) / q) * q
        # we want int values for depth and estimators
        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0, 1)
    }

    # by using functools partial, we create a new function
    # which has same parameters as the optimize function
    # except for the fact that only one param is required
    # this is how gp_minimize expects the optimization function to be.
    # you can change it by reading data inside the optimize function or by 
    # defining the optimize functions here 
    optimization_function = partial(
        optimize,
        x=X,
        y=y
    )

    # init trials to keep logging information
    trials = Trials()

    # run hyperpot
    hpot = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )

    print(hpot)
