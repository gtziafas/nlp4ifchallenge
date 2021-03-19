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


def make_labeled_dataset(path: str) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str) -> List[Tensor]:
    return [tokenize_unlabeled(tweet) for tweet in read_unlabeled(path)]
