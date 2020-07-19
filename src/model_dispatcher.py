from sklearn import ensemble
from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion='gini'
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion='entropy'
    ),
    "rf": ensemble.GradientBoostingClassifier(),
}