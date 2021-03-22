from ...types import *
from .metrics import multi_label_metrics, get_metrics

import torch


def train_epoch(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module, device: str) -> Dict[str, float]:
    model.train()

    epoch_loss = 0.
    all_preds:  List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        
        # forward 
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        all_preds.extend(predictions.sigmoid().round().cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        epoch_loss += loss.item()

    return {**get_metrics(all_preds, all_labels), **{'loss': epoch_loss/len(dl)}}


@torch.no_grad()
def eval_epoch(model: Module, dl: DataLoader, loss_fn: Module, device: str) -> Dict[str, float]:
    model.eval()

    epoch_loss = 0.
    all_preds: List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)

        # forward
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())

        all_preds.extend(predictions.sigmoid().round().cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        epoch_loss += loss.item()

    return {**get_metrics(all_preds, all_labels), **{'loss': epoch_loss / len(dl)}}
