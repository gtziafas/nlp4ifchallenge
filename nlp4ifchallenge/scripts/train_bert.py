from ..types import *
from ..models.bert import *
from ..preprocessing import extract_class_weights
from ..utils.training import Trainer

from torch import manual_seed, save, load
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import AdamW 

from warnings import filterwarnings
import sys 
import os 

SAVE_PREFIX = '/data/s3913171/nlp4ifchallenge/checkpoints'

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
    save_path = '/'.join([save_path, 'model.p'])

    manual_seed(0)
    filterwarnings('ignore')

    model = make_model(name, device)

    train_ds, dev_ds = read_labeled(train_path), read_labeled(dev_path)
    train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size,
                          collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=True)
    dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size,
                          collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=False)

    # if provided test path
    if test_path != '':
        test_ds = read_labeled(test_path)
        test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size,
                          collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=False)

    class_weights = tensor(extract_class_weights(train_path), dtype=floatt, device=device)
    criterion = BCEWithLogitsLoss() if not with_class_weights else BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=3e-05, weight_decay=1e-02)

    trainer = Trainer(model, (train_dl, dev_dl), optimizer, criterion, target_metric='mean_f1', print_log=True)

    best = trainer.iterate(num_epochs, with_save=save_path, with_test=test_dl if test_path != '' else None)
    print(f'Results: {best}')

    # load best saved model and re-save with faiths 
    faiths = array([c['f1'] for c in best['column_wise']])
    save({'faiths': faiths, 'model_state_dict': load(save_path)}, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-tr', '--train_path', help='path to the training data tsv', type=str, default='./data/english/covid19_disinfo_binary_english_train.tsv')
    parser.add_argument('-dev', '--dev_path', help='path to the development data tsv', type=str, default='./data/english/covid19_disinfo_binary_english_dev_input.tsv')
    parser.add_argument('-tst', '--test_path', help='path to the testing data tsv', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=SAVE_PREFIX)
    parser.add_argument('--with_class_weights', action='store_true', help='use pre-computed weights for labels', default=False)

    kwargs = vars(parser.parse_args())
    train_bert(**kwargs)