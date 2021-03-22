from ...types import *
from ...preprocessing import *
from torch import tensor, long, stack
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class BERTLike(Module, Model):
    def __init__(self, name: str, model_dim: int, dropout_rate: float = 0.33):
        super().__init__()
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(model_dim, 7)

    def tensorize_labeled(self, tweets: List[LabeledTweet]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(tweet, self.tokenizer) for tweet in tweets]

    def tensorize_unlabeled(self, tweets: List[Tweet]) -> List[Tensor]:
        return [tokenize_unlabeled(tweet, self.tokenizer) for tweet in tweets]

    def forward(self, x: Tensor):
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict(self, tweets: List[Tweet]) -> List[str]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        preds = self.forward(tensorized).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def collate_tuples(pairs: List[Tuple[Tensor, Tensor]], padding_value: int) -> Tuple[Tensor, Tensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value), stack(ys)


def preds_to_str(preds: List[int]) -> str:
    return '\t'.join(['nan' if preds[0] == 0 and 0 < i < 5 else 'yes' if p == 1 else 'no' for i, p in enumerate(preds)])


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def longt(x: Any) -> Tensor:
    return tensor(x, dtype=long)


def tokenize_text(text: str, tokenizer: AutoTokenizer) -> Tensor:
    return longt(tokenizer.encode(text, truncation=True))


def tokenize_labels(labels: List[Label]) -> Tensor:
    return longt([0 if label is False or label is None else 1 for label in labels])


def tokenize_unlabeled(tweet: Tweet, tokenizer: AutoTokenizer) -> Tensor:
    return tokenize_text(tweet.text, tokenizer)


def tokenize_labeled(tweet: LabeledTweet, tokenizer: AutoTokenizer) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer), tokenize_labels(tweet.labels)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer) for tweet in read_unlabeled(path)]


def make_model(name: str) -> BERTLike:
    if name == 'covid':
        return BERTLike(name='digitalepidemiologylab/covid-twitter-bert-v2', model_dim=1024)
    if name == 'tweet':
        return BERTLike(name='vinai/bertweet-base', model_dim=768)
    else:
        raise ValueError(f'unknown name {name}')
