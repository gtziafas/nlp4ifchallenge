from ..types import *
from ..models.bert import make_model, tokenize_labels
from ..models.aggregation import MetaClassifier
from ..preprocessing import read_unlabeled, read_labeled
from ..utils.training import Trainer
from torch import load, cat, stack, save

from math import ceil

from torch.optim.adamw import AdamW
from torch.nn import BCEWithLogitsLoss

import os
import sys


def sprint(s: Any):
    print(s)
    sys.stdout.flush()


def get_scores(model_names: List[str], datasets: List[List[Tweet]], batch_size: int, device: str,
               model_dir: str) -> List[Tensor]:
    """
       :returns num_dataset tensors of shape B x M x Q
    """
    outs = []
    for name in model_names:
        this_model_outs = []
        model = make_model(name).to(device)
        model.load_state_dict(load(model_dir + name + '/model.p')['model_state_dict'])
        for dataset in datasets:
            this_dataset_outs = []
            nbatches = ceil(len(dataset) / batch_size)
            for batch_idx in range(nbatches):
                start, end = batch_idx * batch_size, min(len(dataset), (batch_idx + 1) * batch_size)
                this_dataset_outs.append(model.predict_scores(dataset[start:end]))
            this_model_outs.append(cat(this_dataset_outs, dim=0))
        outs.append(this_model_outs)
    return [stack(x, dim=1) for x in zip(*outs)]


def _train_aggregator(dls: Tuple[DataLoader, DataLoader], aggregator: MetaClassifier, num_epochs: int,
                      save_dir: str, print_log: bool) -> Dict[str, Any]:
    save_dir = '/'.join([save_dir, 'aggregator'])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir = '/'.join([save_dir, 'model.p'])

    optim = AdamW(aggregator.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = BCEWithLogitsLoss()
    trainer = Trainer(aggregator, dls, optim, criterion, target_metric='mean_f1', print_log=print_log)
    return trainer.iterate(num_epochs, with_save=save_dir)


def simple_collate(pairs: List[Tuple[Tensor, LongTensor]], device: str) -> Tuple[Tensor, LongTensor]:
    xs, ys = list(zip(*pairs))
    return stack(xs).to(device), stack(ys).to(device)


def train(model_names: List[str], train_path: str, dev_path: str, device: str, model_dir: str, batch_size: int,
          num_epochs: int, print_log: bool, load_stored: bool, hidden_size: int):
    def load_scores():
        tmp = load(model_dir + '/scores.p')
        assert tmp['model_names'] == model_names
        return tmp['train_inputs'], tmp['dev_inputs']

    # filter paths
    train_ds = read_labeled(train_path)
    dev_ds = read_labeled(dev_path)

    if not load_stored:
        train_inputs, dev_inputs = get_scores(model_names=model_names, datasets=[train_ds, dev_ds], batch_size=16,
                                               device=device, model_dir=model_dir)
        save({'train_inputs': train_inputs, 'dev_inputs': dev_inputs, 'model_names': model_names},
             model_dir + '/scores.p')
    else:
        train_inputs, dev_inputs = load_scores()

    train_labels = stack([tokenize_labels(s.labels) for s in train_ds], dim=0)
    dev_labels = stack([tokenize_labels(s.labels) for s in dev_ds], dim=0)
    train_dl = DataLoader(list(zip(train_inputs, train_labels)), batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: simple_collate(b, device))
    dev_dl = DataLoader(list(zip(dev_inputs, dev_labels)), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: simple_collate(b, device))
    aggregator = MetaClassifier(num_models=len(model_names), hidden_size=hidden_size).to(device)
    best_metric = _train_aggregator((train_dl, dev_dl), aggregator, num_epochs=num_epochs,
                                    save_dir=model_dir, print_log=print_log)
    sprint(best_metric)


def find_thresholds():
    ...


def test(model_names: List[str], test_path: str, hidden_size: int, device: str,
         model_dir: str, save_to: str) -> List[str]:
    test_ds = read_unlabeled(test_path)
    [test_inputs] = get_scores(model_names, datasets=[test_ds], batch_size=16, device=device, model_dir=model_dir)
    aggregator = MetaClassifier(num_models=len(model_names), hidden_size=hidden_size).to(device)
    aggregator.load_state_dict(load(model_dir + '/aggregator/model.p'))
    outs = aggregator.threshold(test_inputs.to(device))
    with open(save_to, 'w') as f:
        f.write('\n'.join(outs))
    return outs


