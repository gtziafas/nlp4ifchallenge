from ..types import *
from ..preprocessing import *
from .utils.metrics import preds_to_str
from .training import train_epoch, eval_epoch
from torch import tensor, long, stack, manual_seed, save
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch.optim import AdamW 

from transformers import AutoModel, AutoTokenizer
from warnings import filterwarnings
import sys
import os

SAVE_PREFIX = '/data/s3913171'


def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


class BERTLike(Module, Model):
    def __init__(self, name: str, model_dim: int = 768, dropout_rate: float = 0.5, 
            max_length: Maybe[int] = None, token_name: Maybe[str] = None):
        super().__init__()
        self.token_name = name if token_name is None else token_name
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name, use_fast=False)
        self.max_length = max_length
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(model_dim, 7)

    def tensorize_labeled(self, tweets: List[LabeledTweet]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(tweet, self.tokenizer, max_length=self.max_length) for tweet in tweets]

    def tensorize_unlabeled(self, tweets: List[Tweet]) -> List[Tensor]:
        return [tokenize_unlabeled(tweet, self.tokenizer, max_length=self.max_length) for tweet in tweets]

    def forward(self, x: Tensor):
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict_scores(self, tweets: List[Tweet]) -> List[List[float]]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        return self.forward(tensorized).sigmoid().cpu().tolist()

    def predict(self, tweets: List[Tweet]) -> List[str]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        preds = self.forward(tensorized).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def collate_tuples(pairs: List[Tuple[Tensor, Tensor]], padding_value: int) -> Tuple[Tensor, Tensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value), stack(ys)


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def longt(x: Any) -> Tensor:
    return tensor(x, dtype=long)


def tokenize_text(text: str, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return longt(tokenizer.encode(text, truncation=True, **kwargs))


def tokenize_labels(labels: List[Label]) -> Tensor:
    return longt([0 if label is False or label is None else 1 for label in labels])


def tokenize_unlabeled(tweet: Tweet, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tokenize_text(tweet.text, tokenizer, **kwargs)


def tokenize_labeled(tweet: LabeledTweet, tokenizer: AutoTokenizer, **kwargs) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer, **kwargs), tokenize_labels(tweet.labels)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer, **kwargs) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer, **kwargs) for tweet in read_unlabeled(path)]


def make_model(name: str) -> BERTLike:
    # todo: find all applicable models
    if name == 'del-covid':
        return BERTLike(name='digitalepidemiologylab/covid-twitter-bert-v2', model_dim=1024)
    elif name == 'vinai-covid':
        return BERTLike(name='vinai/bertweet-covid19-base-cased', max_length=128)
    elif name == 'vinai-tweet':
        return BERTLike(name='vinai/bertweet-base', max_length=128)
    elif name == 'cardiffnlp-tweet':
        return BERTLike(name='cardiffnlp/twitter-roberta-base')
    elif name == 'cardiffnlp-sentiment':
        return BERTLike(name='cardiffnlp/twitter-roberta-base-sentiment')
    elif name == 'cardiffnlp-hate':
        return BERTLike(name='cardiffnlp/twitter-roberta-base-hate')
    elif name == 'cardiffnlp-emotion':
        return BERTLike(name='cardiffnlp/twitter-roberta-base-emotion')
    elif name == 'cardiffnlp-offensive':
        return BERTLike(name='cardiffnlp/twitter-roberta-base-offensive')
    elif name == 'cardiffnlp-irony':
        return BERTLike(name='cardiffnlp/twitter-roberta-base-irony')
    
    # multi-lingual models
    elif name == 'multi-bert':
        return BERTLike(name='bert-base-multilingual-cased')
    elif name == 'multi-xlm':
        return BERTLike(name='xlm-roberta-base')
    elif name == 'multi-xnli':
        return BERTLike(name='joeddav/xlm-roberta-large-xnli', model_dim=1024)
    elif name == 'multi-microsoft':
        return BERTLike(name='microsoft/Multilingual-MiniLM-L12-H384', model_dim=384, token_name='xlm-roberta-base')
    elif name == 'multi-sentiment':
        return BERTLike(name='socialmediaie/TRAC2020_ALL_C_bert-base-multilingual-uncased')
    elif name == 'multi-toxic':
        return BERTLike(name='unitary/multilingual-toxic-xlm-roberta')

    else:
        raise ValueError(f'unknown name {name}')


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

    class_weights = tensor([0.6223021582733813, 6.151515151515151, 0.2328767123287671, 1.4210526315789473, 0.8783783783783784, 3.4455445544554455, 1.6023391812865497], dtype=torch.float, device=device)
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
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=32)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/nlp4ifchallenge/checkpoints')
    parser.add_argument('--with_class_weights', action='store_true', help='use pre-computed weights for labels', default=False)

    kwargs = vars(parser.parse_args())
    train_bert(**kwargs)