import os

# fetch home dir
HOME_DIR = os.path.dirname(os.path.realpath('__file__'))

### mnistdata ###
TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\mnist_train_sample.csv")
TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\mnist_train_folds.csv")
TESTING_FILE = os.path.join(HOME_DIR, ".\\input\\mnist_test.csv")
MODEL_OUTPUT = os.path.join(HOME_DIR, ".\\models")

### cat data ###
CAT_TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\cat_train.csv")
CAT_TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\cat_train_folds.csv")
CAT_TESTING_FILE = os.path.join(HOME_DIR, ".\\input\\cat_test.csv")

### census data ###
CEN_TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\CensusData.csv")
CEN_TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\cen_train_folds.csv")

### bless data ###
BLESS_TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\bless_orig_sample.csv")
BLESS_TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\bless_orig_sample_folds.csv")

### mobile train data ###
MOB_TRAIN_FILE = os.path.join(HOME_DIR, ".\\input\\mobile_train.csv")
MOB_TRAIN_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\mobile_train_folds.csv")

### dep data ###
DEP_TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\dep_train.csv")
DEP_TESTING_FILE = os.path.join(HOME_DIR, ".\\input\\dep_test.csv")

