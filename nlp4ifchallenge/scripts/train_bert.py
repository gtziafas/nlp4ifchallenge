from ..types import *
from ..models.bert import *
from ..utils.training import train_epoch, eval_epoch

from torch import manual_seed, save 
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import AdamW 

from warnings import filterwarnings
import sys 
import os 

SAVE_PREFIX = '/data/s3913171/nlp4ifchallenge/checkpoints'


def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


def train_bert(name: str,
               train_path: str,
               dev_path: str,
               test_path: str,
               device: str,
               batch_size: int,
               num_epochs: int,
               save_path: str,
               with_class_weights: bool):
    save_path = '/'.join([save_path, name])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    manual_seed(0)
    filterwarnings('ignore')

    model = make_model(name).to(device)

    train_ds, dev_ds = read_labeled(train_path), read_labeled(dev_path)
    train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=True)
    dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=False)

    # if provided test path 
    if test_path != '':
        test_ds = read_labeled(test_path)
        test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size,
                          collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=False)

    class_weights = tensor([
        0.6223021582733813, 
        6.151515151515151, 
        0.2328767123287671, 
        1.4210526315789473, 
        0.8783783783783784, 
        3.4455445544554455, 
        1.6023391812865497], dtype=floatt, device=device)
    #class_weights = tensor([0.6223, 12.6667,  1.0594,  2.9561,  2.0473,  3.4653,  1.6374], device=device)
    #criterion = BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = BCEWithLogitsLoss() if not with_class_weights else BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-05, weight_decay=1e-02, print_change_log=False)

    train_log, dev_log, test_log = [], [], []
    best = 0.
    for epoch in range(num_epochs):
        train_log.append(train_epoch(model, train_dl, optimizer, criterion, device))
        sprint(train_log[-1])
        dev_log.append(eval_epoch(model, dev_dl, criterion, device))
        sprint(dev_log[-1])
        sprint('=' * 64)
        mean_f1 = dev_log[-1]['mean_f1']
        if mean_f1 > best:
            best = mean_f1
            faith = array([c['f1'] for c in dev_log[-1]['column_wise']])
            save(
                {'faith': faith, 'model_state_dict': model.state_dict()}, f'{save_path}/model.p')
            # eval on test set for each new best model
            if test_path != '':
                test_log.append(eval_epoch(model, test_dl, criterion, device))
                sprint('\nTEST\n')
                sprint(test_log[-1])
                sprint('=' * 64)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-tr', '--train_path', help='path to the training data tsv', type=str, default='./nlp4ifchallenge/data/english/covid19_disinfo_binary_english_train.tsv')
    parser.add_argument('-dev', '--dev_path', help='path to the development data tsv', type=str, default='./nlp4ifchallenge/data/english/covid19_disinfo_binary_english_dev_input.tsv')
    parser.add_argument('-tst', '--test_path', help='path to the testing data tsv', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=SAVE_PREFIX)
    parser.add_argument('--with_class_weights', action='store_true', help='use pre-computed weights for labels', default=False)

    kwargs = vars(parser.parse_args())
    train_bert(**kwargs)