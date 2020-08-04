import nlp_config
import torch


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        # for any given item, which is an int,
        # return review and targets as torch tensor 
        # item is the index of the item in concern
        review = self.reviews[item, :]
        target = self.target[item]

        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }


class BERTDataset:
    def __init__(self, review, target):
        """
        :param review: list or numpy array of strings
        :param target: list or numpy array which is binary
        """
        self.review = review
        self.target = target
        self.tokenizer = nlp_config.TOKENIZER
        self.max_len = nlp_config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        # for a given item index, return a dictionary
        # of inputs
        review = str(self.review[item])
        review = " ".join(review.split())

        # encode_plus comes from huggingface's transformers
        # and exists for all tokenizers they offer
        # it can be used to convert a given string to ids,
        # mask and token type ids which are needed for models like BERT
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        # ids are ids of tokens genrated 
        # after tokenizing reviews
        ids = inputs["input_ids"]
        # mask is 1 where we have input
        # and 0 where we have padding
        mask = input["attention_mask"]
        # token type ids behave the same way as
        # mask in this specific case
        # in case of two sentences, this is 0
        # for first sentence and 1 for second sentence
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(
                ids, dtype=torch.long
            ),
            "mask": torch.tensor(
                mask, dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                token_type_ids, dtype=torch.long
            ),
            "targets": torch.tensor(
                self.target[item], dtype=torch.float
            )
        }
