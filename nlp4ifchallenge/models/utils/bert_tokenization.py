from ...types import *
from ...preprocessing import *
from transformers import BertTokenizer
from torch import tensor, long, stack
from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def collate_tuples(pairs: List[Tuple[Tensor, Tensor]], padding_value: int) -> Tuple[Tensor, Tensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value), stack(ys)


def preds_to_str(preds: List[int]) -> str:
    return '\t'.join(['nan' if preds[0] == 0 and 0 < i < 5 else 'yes' if p == 1 else 'no' for i, p in enumerate(preds)])


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def longt(x: Any) -> Tensor:
    return tensor(x, dtype=long)


def tokenize_text(text: str, tokenizer: BertTokenizer) -> Tensor:
    return longt(tokenizer.encode(text))


def tokenize_labels(labels: List[Label]) -> Tensor:
    return longt([0 if label is False or label is None else 1 for label in labels])


def tokenize_unlabeled(tweet: Tweet, tokenizer: BertTokenizer) -> Tensor:
    return tokenize_text(tweet.text, tokenizer)


def tokenize_labeled(tweet: LabeledTweet, tokenizer: BertTokenizer) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer), tokenize_labels(tweet.labels)


def make_labeled_dataset(path: str, tokenizer: BertTokenizer) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: BertTokenizer) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer) for tweet in read_unlabeled(path)]
