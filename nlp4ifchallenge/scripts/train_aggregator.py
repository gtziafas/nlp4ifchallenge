from ..types import *
from ..models.bert import make_model, tokenize_labels
from ..models.aggregation import MetaClassifier
from ..preprocessing import read_unlabeled, read_labeled
from ..utils.training import Trainer

from math import ceil

from torch.optim.adamw import AdamW
from torch.cuda import empty_cache
from torch.nn import BCEWithLogitsLoss
from torch import load, cat, stack, save, no_grad

import os
import sys

MODELS_ENSEMBLE = ['vinai-covid', 'vinai-tweet', 'cardiffnlp-offensive']
SAVE_PREFIX = '/data/s3913171/nlp4ifchallenge/checkpoints'


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
        model = make_model(name, True).to(device)
        model.load_state_dict(load('/'.join([model_dir, name, 'model.p']))['model_state_dict'])
        for dataset in datasets:
            this_dataset_outs = []
            nbatches = ceil(len(dataset) / batch_size)
            for batch_idx in range(nbatches):
                start, end = batch_idx * batch_size, min(len(dataset), (batch_idx + 1) * batch_size)
                this_dataset_outs.append(model.predict_scores(dataset[start:end]))
            this_model_outs.append(cat(this_dataset_outs, dim=0))
        outs.append(this_model_outs)
        empty_cache()
    return [stack(x, dim=1) for x in zip(*outs)]


def simple_collate(pairs: List[Tuple[Tensor, LongTensor]], device: str) -> Tuple[Tensor, LongTensor]:
    xs, ys = list(zip(*pairs))
    return stack(xs).to(device), stack(ys).to(device)


def train(model_names: List[str], train_path: str, dev_path: str, device: str, model_dir: str, batch_size: int,
          num_epochs: int, print_log: bool, load_stored: bool, hidden_size: int):
    save_dir = '/'.join([model_dir, 'aggregator'])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir = '/'.join([save_dir, 'model.p'])

    def load_scores():
        tmp = load(model_dir + '/scores.p')
        assert tmp['model_names'] == model_names
        return tmp['train_inputs'], tmp['dev_inputs']


    # load data from checkpoint if extracted scores once
    train_ds, dev_ds = read_labeled(train_path), read_labeled(dev_path)
    if not load_stored:
        train_inputs, dev_inputs = get_scores(model_names=model_names, datasets=[train_ds, dev_ds], batch_size=16,
                                               device=device, model_dir=model_dir)
        save({'train_inputs': train_inputs, 'dev_inputs': dev_inputs, 'model_names': model_names},
             model_dir + '/scores.p')
    else:
        train_inputs, dev_inputs = load_scores()

    train_labels = stack([tokenize_labels(s.labels, True) for s in train_ds], dim=0)
    dev_labels = stack([tokenize_labels(s.labels, True) for s in dev_ds], dim=0)
    train_dl = DataLoader(list(zip(train_inputs, train_labels)), batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: simple_collate(b, device))
    dev_dl = DataLoader(list(zip(dev_inputs, dev_labels)), batch_size=batch_size, shuffle=False,
                          collate_fn=lambda b: simple_collate(b, device))

    model = MetaClassifier(num_models=len(model_names), hidden_size=hidden_size).to(device)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = BCEWithLogitsLoss()
    trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric='mean_f1', print_log=print_log)
    return trainer.iterate(num_epochs, with_save=save_dir)


def find_thresholds():
    ...


def test(model_names: List[str], test_path: str, hidden_size: int, device: str, model_dir: str, save_to: str) -> List[str]:
    test_ds = read_unlabeled(test_path)
    [test_inputs] = get_scores(model_names, datasets=[test_ds], batch_size=16, device=device, model_dir=model_dir)
    aggregator = MetaClassifier(num_models=len(model_names), hidden_size=hidden_size).to(device)
    aggregator.load_state_dict(load(model_dir + '/aggregator/model.p'))
    outs = aggregator.threshold(test_inputs.to(device))
    with open(save_to, 'w') as f:
        f.write('\n'.join(outs))


def main(model_names: List[str], train_path: str, dev_path: str, device: str, model_dir: str, batch_size: int,
          test_path: str, num_epochs: int, print_log: bool, load_stored: bool, hidden_size: int):
    #model_names = model_names.split(',')
    sprint(model_names)
    
    # if test path is given do only testing
    if test_path != '':
        out_file = model_dir + '/test.out'
        test(model_names, test_path, hidden_size, device, model_dir, save_to=out_file)
        return

    # otherwise we train the aggregator
    best = train(model_names, train_path, dev_path, device, model_dir, batch_size, num_epochs, print_log, load_stored,
                 hidden_size)

    sprint(f'Results: {best}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_names', help='names of BERT models for ensemble (given as ,-seperated str)', type=List[str], default=MODELS_ENSEMBLE)
    parser.add_argument('-tr', '--train_path', help='path to the training data tsv', type=str, default='./data/english/covid19_disinfo_binary_english_train.tsv')
    parser.add_argument('-dev', '--dev_path', help='path to the development data tsv', type=str, default='./data/english/covid19_disinfo_binary_english_dev_input.tsv')
    parser.add_argument('-tst', '--test_path', help='path to the testing data tsv', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=20)
    parser.add_argument('-s', '--model_dir', help='prefix to load model paths', type=str, default=SAVE_PREFIX)
    parser.add_argument('-dh', '--hidden_size', help='size of meta-classifier hidden layer', type=int, default=14)
    parser.add_argument('--load_stored', action='store_true', help='whether to load scores from checkpoint', default=False)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)