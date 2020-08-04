import nlp_config
import flask
import time
import torch
import torch.nn as nn
from flask import request
from flask import Flask
from model import BERTBaseUncased

app = Flask(__name__)

MODEL = None
DEVICE = "cuda"

def sentence_prediction(sentence):
    """
    A prediction function that takes an input sentence
    and returns the probability for it being associated 
    to postive sentiment
    """
    tokenizer = nlp_config.TOKENIZER
    max_len = nlp_config.MAX_LEN

    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_token=True,
        max_length=max_len
    )

    ids = inputs["inputs_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    return outputs[0][0]

@app.route("/predict", methods=["GET"])
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction

    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time)
    }

    return flask.jsonify(response)

if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(
        nlp_config.MODEL_PATH, map_location=torch.device(DEVICE)
    ))

    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="0.0.0.0")