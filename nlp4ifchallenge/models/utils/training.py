from ..types import * 
from .metrics import multi_label_metrics

import torch


def train_epoch(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module, device: str) -> Tuple[float, float, float]:
    model.train()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        
        # forward 
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())
        metrics = multi_label_metrics(predictions, y)

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        batch_loss += loss.item()
        batch_accu += metrics['accuracy']
        batch_hamm_loss += metrics['hamming_loss']

    return batch_loss/len(dl), batch_accu/len(dl), batch_hamm_loss/len(dl) 


@torch.no_grad()
def eval_epoch(model: Module, dl: DataLoader, loss_fn: Module, device: str) -> Tuple[float, float, float]:
    model.eval()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())
        metrics = multi_label_metrics(predictions, y)
        batch_loss += loss.item()
        batch_accu += metrics['accuracy']
        batch_hamm_loss += metrics['hamming_loss']

    return batch_loss/len(dl), batch_accu/len(dl), batch_hamm_loss/len(dl) 