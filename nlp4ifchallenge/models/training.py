from ..types import *
from .utils.metrics import get_metrics
from .bert import *
from adabelief_pytorch import AdaBelief
from warnings import filterwarnings
from sklearn.utils.class_weight import compute_class_weight

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

    return {'loss': round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}


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

    return {'loss': round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}


def train_bert(name: str,
               train_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_train.tsv',
               dev_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_dev_input.tsv',
               test_path: str = '',
               device: str = 'cuda',
               batch_size: int = 1,
               num_epochs: int = 10):
    save_path = f'./nlp4ifchallenge/checkpoints/{name}'

    torch.manual_seed(0)
    filterwarnings('ignore')

    model = make_model(name).to(device)

    train_ds, dev_ds = read_labeled(train_path), read_labeled(dev_path)
    train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=True)
    dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=False)

    class_weights = tensor([0.6223, 12.6667,  1.0594,  2.9561,  2.0473,  3.4653,  1.6374], device=device)
    criterion = BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = AdaBelief(model.parameters(), lr=1e-05, weight_decay=1e-01, print_change_log=False)

    train_log, dev_log = [], []
    best = 0.
    for epoch in range(num_epochs):
        train_log.append(train_epoch(model, train_dl, optimizer, criterion, device))
        print(train_log[-1])
        dev_log.append(eval_epoch(model, dev_dl, criterion, device))
        print(dev_log[-1])
        print('=' * 64)
        mean_f1 = dev_log[-1]['mean_f1']
        if mean_f1 > best:
            best = mean_f1
            torch.save(model.state_dict(), f'{save_path}/model.p')
