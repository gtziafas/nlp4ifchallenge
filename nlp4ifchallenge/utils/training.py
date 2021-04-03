from ..types import *
from ..utils.metrics import get_metrics

from torch.nn import Module
import torch


def train_epoch(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module) -> Dict[str, Any]:
    model.train()

    epoch_loss = 0.
    all_preds:  List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
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

    return {'loss': round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}


@torch.no_grad()
def eval_epoch(model: Module, dl: DataLoader, loss_fn: Module) -> Dict[str, Any]:
    model.eval()

    epoch_loss = 0.
    all_preds: List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
        # forward
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())

        all_preds.extend(predictions.sigmoid().round().cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        epoch_loss += loss.item()

    return {'loss': round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}