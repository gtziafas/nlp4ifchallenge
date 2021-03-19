from ...types import *
from ...prepocessing import *
from transformers import BertTokenizer
from torch import tensor, long

tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')


def longt(x: Any) -> Tensor:
    return tensor(x, dtype=long)


def tokenize_text(text: str) -> Tensor:
    return longt(tokenizer.encode(text))


def tokenize_labels(labels: List[Label]) -> Tensor:
    return longt([0 if label is None else 1 if label else False for label in labels])


def tokenize_unlabeled(tweet: Tweet) -> Tensor:
    return tokenize_text(tweet.text)


def tokenize_labeled(tweet: LabeledTweet) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet), tokenize_labels(tweet.labels)


@overload
def make_dataset(path: str, labeled: Literal[True]) -> List[Tuple[Tensor, Tensor]]:
    pass


@overload
def make_dataset(path: str, labeled: Literal[False]) -> List[Tensor]:
    pass


def make_dataset(path, labeled):
    read_fn = read_labeled if labeled else read_labeled
    tok_fn = tokenize_labeled if labeled else tokenize_unlabeled
    return [tok_fn(tweet) for tweet in read_fn(path)]
