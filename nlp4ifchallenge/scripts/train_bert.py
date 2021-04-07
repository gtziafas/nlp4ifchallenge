from ..types import *
from ..models.bert import *
from ..utils.training import Trainer
from ..utils.loss import BCEWithLogitsIgnore, BCEWithLogitsLoss
from ..preprocessing import read_labeled, extract_class_weights

from torch import manual_seed, save, load
from torch.optim import AdamW 

from warnings import filterwarnings
import sys 
import os 

SAVE_PREFIX = '/data/s3913171/nlp4ifchallenge/checkpoints'
manual_seed(0)
filterwarnings('ignore')


def sprint(s: str):
    print(s)
    sys.stdout.flush()


def main(name: str,
        train_path: str,
        dev_path: str,
        test_path: str,
        device: str,
        batch_size: int,
        early_stopping: Maybe[int],
        num_epochs: int,
        save_path: str,
        print_log: bool,
        with_class_weights: bool,
        ignore_nan: bool):
    data_tag = train_path.split('data')[1].split('/')[1]
    save_path = '/'.join([save_path, '-'.join([name, data_tag])])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = '/'.join([save_path, 'model.p'])

    model = make_model(name, ignore_nan).to(device)

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
    criterion = BCEWithLogitsLoss(pos_weight=class_weights) if with_class_weights else BCEWithLogitsIgnore(ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=3e-05, weight_decay=1e-02)

    trainer = Trainer(model, (train_dl, dev_dl), optimizer, criterion, target_metric='mean_f1', early_stopping=early_stopping, print_log=print_log)

    best = trainer.iterate(num_epochs, with_save=save_path, with_test=test_dl if test_path != '' else None)
    sprint(f'{name}: {best}')
    if test_path != '':
        sprint(f'\nbest test -- {trainer.logs["test"][-1]}')
    
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
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=20)
    parser.add_argument('-early', '--early_stopping', help='early stopping patience (default no)', type=int, default=0)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=SAVE_PREFIX)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('--with_class_weights', action='store_true', help='compute class weights for loss penalization', default=False)
    parser.add_argument('--ignore_nan', action='store_true', help='set True to ignore (not penalize) nan labels', default=False)

    kwargs = vars(parser.parse_args())
    main(**kwargs)