import os
import transformers

# fetch home dir
HOME_DIR = os.path.dirname(os.path.realpath('__file__'))

### mnistdata ###
TRAINING_FILE = os.path.join(HOME_DIR, ".\\input\\imdb_data.csv")
TRAINING_FILE_FOLDS = os.path.join(HOME_DIR, ".\\input\\imdb_data_folds.csv")
MODEL_OUTPUT = os.path.join(HOME_DIR, ".\\models")

# this is the maximum number of tokens in the sentence
MAX_LEN = 128

# batch sizes is small because model is huge
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8

# train on maximum 10 epochs
EPOCHS = 10

# define path to BERT model files
# assume all data is here /home/{user}/data
BERT_PATH = os.path.join(HOME_DIR, ".\\input\\bert_base_uncased\\")

# this is where you want to save the model
MODEL_PATH = os.path.join(HOME_DIR, ".\\models\\model.bin")

# define the tokenizer
# we use tokenizer and model
# from huggingface's trasformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)

DEVICE = "cude"


