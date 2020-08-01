import torch


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews