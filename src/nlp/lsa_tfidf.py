from nltk import corpus
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

import nlp_config
from tabular.create_folds import feature


def clean_text(s):
    """
    This functions clean the text a bit
    :param s: string
    :return: cleaned string
    """
    s = s.split()
    # join token by single space
    # why we do this?
    # this will remove all kinds of weird space
    # "hi.    how are you" becomes
    # "hi. how are you"
    s = " ".join(s)

    # remove all punctuations using regex and string module
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)

    return s

if __name__ == "__main__":
    corpus = pd.read_csv(nlp_config.TRANING_FILE)
    corpus.loc[:, "review"] = corpus.review.apply(clean_text)
    corpus = corpus.review.values

    tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    tfv.fit(corpus)
    corpus_transformed = tfv.transform(corpus)

    svd = decomposition.TruncatedSVD(n_components=10)
    corpus_svd = svd.fit(corpus_transformed)

    # choose first sample and create a dictionary
    # of feature names and their scores from svd
    # we can change the sample_index variable to
    # get dict for any other sample
    sample_index = 0
    N = 5

    for sample_index in range(5):
        feature_scores = dict(
            zip(
                tfv.get_feature_names(),
                corpus_svd.components_[sample_index]
            )
        )
        # once we have the dictionary, we can now
        # sort it in decreasing order and get the 
        # top N topics
        print(
            sorted(
                feature_scores,
                key=feature_scores.get,
                reverse=True
            )
        )

