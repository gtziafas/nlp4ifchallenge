from ...types import *
from .metrics import get_metrics
from .bert import *
from adabelief_pytorch import AdaBelief

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


def train_bert(name: str,
               train_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_train.tsv',
               dev_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_dev_input.tsv',
               test_path: str = '',
               device: str = 'cuda',
               batch_size: int = 1):
    # todo
    torch.manual_seed(0)

    model = make_model(name).to(device)

    train_ds = read_labeled(train_path)
    train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id))

    criterion = BCEWithLogitsLoss()
    optimizer = AdaBelief(model.parameters(), lr=1e-05, weight_decay=1e-02, print_change_log=False)

    num_epochs = 5
    log: List[Dict] = []
    for epoch in range(num_epochs):
        log.append(train_epoch(model, train_dl, optimizer, criterion, device))
        print(log[-1])
