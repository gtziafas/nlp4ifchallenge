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
        self.faith = array([0.] * 7)

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
