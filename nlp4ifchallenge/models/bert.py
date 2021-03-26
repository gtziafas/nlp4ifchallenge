from ..types import *
from ..preprocessing import *
from torch import tensor, long, stack
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from .utils.metrics import preds_to_str

from transformers import AutoModel, AutoTokenizer


class BERTLike(Module, Model):
    def __init__(self, name: str, model_dim: int, dropout_rate: float = 0.5, max_length: Maybe[int] = None):
        super().__init__()
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
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
    if name == 'vinai-covid':
        return BERTLike(name='vinai/bertweet-covid19-base-cased', model_dim=768, max_length=128)
    if name == '':
        pass
    else:
        raise ValueError(f'unknown name {name}')


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
            faith = array([c['f1'] for c in dev_log[-1]['column_wise']])
            torch.save(
                {'faith': faith, 'model_state_dict': model.state_dict()}, f'{save_path}/model.p')
