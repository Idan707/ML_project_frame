import torch
import torch.nn as nn
import tqdm

def loss_fn(outputs, targets):
    """
    This function returns the loss
    :param outputs: output from the model (real numbers)
    :param targets: input targets (binary)
    """
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This is the main training function that trains model
    for one epoch
    :param model: model (bert model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param device: this can be "cuda" or "cpu"
    :param scheduler: learning rate scheduler
    """
    model.train()

    # go through batches of data in data loader
    for d in data_loader:
        # fetch ids, token type ids and mask
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        # move the data to device that we want to use
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # pass through the model
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # calculate the loss
        loss = loss_fn(outputs, targets)

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()

        # single step scheduler
        scheduler.step()

def eval_fn(data_loader, model, device):
    """
    This is the validation function that generates 
    predictions on validation data
    :param data_loader: torch dataloader object
    :param model: torch model, bert in our case
    :param device: can be cpu or cuda
    :return: output and targets
    """
    # put the model in eval mode
    model.eval()

    # init empty lists to store preds and targets
    fin_targets = []
    fin_outputs = []

    # disable gradient calculation
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            targets = targets.cpu().detach()
            fin_targets.extend(targets.numpy().tolist())

            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())


    return fin_outputs, fin_targets

def train(data_loader, model, optimizer, device):
    """
    This is the main training function that trains model
    for one epoch
    :param model: model (lstm model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param device: this can be "cuda" or "cpu"
    """
    model.train()

    # go through batches of data in data loader
    for data in data_loader:
        # fetch review and target from the dict
        reviews = data["review"]
        targets = data["target"]

        # move the data to device that we want to use
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make preds
        predictions = model(reviews)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1, 1)
        )

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()

def evaluate(data_loader, model, device):
    # init empty lists to store preds and targets
    final_predictions = []
    final_targets = []

    # put the model in eval mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # make predictions
            predictions = model(reviews)

            # move preds and targets to list
            # we need to move preds and targets to cpu too
            predictions = predictions.cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets
