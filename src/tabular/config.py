import os

# fetch home dir
HOME_DIR = os.path.expanduser("~")

#TRAINING_FILE = os.path.join(HOME_DIR, "git", "ML_project_frame", "input", "mnist_train.csv")
TRAINING_FILE = os.path.join(HOME_DIR, "Documents","GitHub", "ML_project_frame", "input", "mnist_train.csv")

#TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, "git", "ML_project_frame", "input", "mnist_train_folds.csv")
TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, "Documents","GitHub", "ML_project_frame", "input", "mnist_train_folds.csv")

TESTING_FILE = os.path.join(HOME_DIR, "git", "ML_project_frame", "input", "mnist_test.csv")

MODEL_OUTPUT = os.path.join(HOME_DIR, "git", "ML_project_frame", "models")