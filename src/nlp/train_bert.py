from scipy.sparse.construct import random
from torch import device
import nlp_config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def train():
    dfx = pd.read_csv(nlp_config.TRAINING_FILE).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # init BERTDataset from dataset.py - train
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=nlp_config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    # init BERTDataset from dataset.py - valid
    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=nlp_config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # init cuda device, use cpu if you dont have GPU
    device = torch.device(nlp_config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    # params we want to optimize
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # calc number of training steps, used by the scheduler
    num_train_steps = int(
        len(df_train) / nlp_config.TRAIN_BATCH_SIZE * nlp_config.EPOCHS
    )

    optimizer = AdamW(optimizer_parameters)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # if we have multiple GPUs
    # model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(nlp_config.EPOCHS):
        engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler
        )
        outputs, targets = engine.eval_fn(
            valid_data_loader, model, device
        )
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), nlp_config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    train()