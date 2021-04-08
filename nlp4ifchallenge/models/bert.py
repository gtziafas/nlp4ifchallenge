from ..types import *
from ..utils.metrics import preds_to_str
from ..preprocessing import read_labeled, read_unlabeled

from torch import tensor, stack, no_grad
from torch.nn import Module, Linear, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class BERTLike(Module, Model):
    def __init__(self, name: str, model_dim: int = 768, dropout_rate: float = 0.5,
                 num_classes: int = 7, max_length: Maybe[int] = None, token_name: Maybe[str] = None,
                 ignore_nan: bool = False):
        super().__init__()
        self.token_name = name if token_name is None else token_name
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name, use_fast=False)
        self.max_length = max_length
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(model_dim, num_classes)
        self.ignore_nan = ignore_nan

    def tensorize_labeled(self, tweets: List[LabeledTweet]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(tweet, self.tokenizer, ignore_nan=self.ignore_nan, max_length=self.max_length)
                for tweet in tweets]

    def tensorize_unlabeled(self, tweets: List[Tweet]) -> List[Tensor]:
        return [tokenize_unlabeled(tweet, self.tokenizer, max_length=self.max_length) for tweet in tweets]

    def forward(self, x: Tensor) -> Tensor:
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    @no_grad()
    def last_hidden_state(self, tweets: List[Tweet]) -> Tensor:
        x = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id).to(self.core.device)
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        h, _ = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return h

    @no_grad()
    def predict_scores(self, tweets: List[Tweet]) -> Tensor:
        self.eval()
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets),
                                  padding_value=self.tokenizer.pad_token_id).to(self.core.device)
        return self.forward(tensorized).sigmoid()

    def predict(self, tweets: List[Tweet], threshold: float = 0.5) -> List[str]:
        preds = self.predict_scores(tweets).ge(threshold).long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def collate_tuples(pairs: List[Tuple[Tensor, LongTensor]], padding_value: int, device: str) -> Tuple[Tensor, LongTensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value).to(device), stack(ys).to(device)


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def tokenize_text(text: str, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tensor(tokenizer.encode(text, truncation=True, **kwargs), dtype=longt)


def tokenize_labels(labels: List[Label], ignore_nan: bool) -> LongTensor:
    none_label = 0 if not ignore_nan else -1
    return tensor([none_label if label is None else 0 if label is False else 1 for label in labels], dtype=longt)


def tokenize_unlabeled(tweet: Tweet, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tokenize_text(tweet.text, tokenizer, **kwargs)


def tokenize_labeled(tweet: LabeledTweet, tokenizer: AutoTokenizer,
                     ignore_nan: bool = False, **kwargs) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer, **kwargs), tokenize_labels(tweet.labels, ignore_nan)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer, **kwargs) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer, **kwargs) for tweet in read_unlabeled(path)]


def make_model(name: str, ignore_nan: bool) -> BERTLike:
    if name == 'del-covid':
        return BERTLike('digitalepidemiologylab/covid-twitter-bert-v2', model_dim=1024, ignore_nan=ignore_nan)
    elif name == 'vinai-covid':
        return BERTLike('vinai/bertweet-covid19-base-cased', max_length=128, ignore_nan=ignore_nan)
    elif name == 'vinai-tweet':
        return BERTLike('vinai/bertweet-base', max_length=128, ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-tweet':
        return BERTLike('cardiffnlp/twitter-roberta-base', ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-sentiment':
        return BERTLike('cardiffnlp/twitter-roberta-base-sentiment', ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-hate':
        return BERTLike('cardiffnlp/twitter-roberta-base-hate', ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-emotion':
        return BERTLike('cardiffnlp/twitter-roberta-base-emotion', ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-offensive':
        return BERTLike('cardiffnlp/twitter-roberta-base-offensive', ignore_nan=ignore_nan)
    elif name == 'cardiffnlp-irony':
        return BERTLike('cardiffnlp/twitter-roberta-base-irony', ignore_nan=ignore_nan)
    
    # multi-lingual models
    elif name == 'multi-bert':
        return BERTLike('bert-base-multilingual-cased', ignore_nan=ignore_nan)
    elif name == 'multi-xlm':
        return BERTLike('xlm-roberta-base', ignore_nan=ignore_nan)
    elif name == 'multi-xnli':
        return BERTLike('joeddav/xlm-roberta-large-xnli', model_dim=1024, ignore_nan=ignore_nan)
    elif name == 'multi-microsoft':
        return BERTLike('microsoft/Multilingual-MiniLM-L12-H384', model_dim=384,
                        token_name='xlm-roberta-base', ignore_nan=ignore_nan)
    elif name == 'multi-sentiment':
        return BERTLike('socialmediaie/TRAC2020_ALL_C_bert-base-multilingual-uncased', ignore_nan=ignore_nan)
    elif name == 'multi-toxic':
        return BERTLike('unitary/multilingual-toxic-xlm-roberta', ignore_nan=ignore_nan)

    else:
        raise ValueError(f'unknown name {name}')