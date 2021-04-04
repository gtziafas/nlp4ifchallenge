from ..types import *
from ..preprocessing import *
from ..utils.metrics import preds_to_str

from torch import tensor, stack
from torch.nn import Module, Linear, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class BERTLike(Module, Model):
    def __init__(self, name: str, device: str, model_dim: int = 768, dropout_rate: float = 0.5, 
            num_classes: int = 7, max_length: Maybe[int] = None, token_name: Maybe[str] = None):
        super().__init__()
        self.device = device
        self.token_name = name if token_name is None else token_name
        self.core = AutoModel.from_pretrained(name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name, use_fast=False)
        self.max_length = max_length
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(model_dim, num_classes)

    def tensorize_labeled(self, tweets: List[LabeledTweet]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(tweet, self.tokenizer, self.device, max_length=self.max_length) for tweet in tweets]

    def tensorize_unlabeled(self, tweets: List[Tweet]) -> List[Tensor]:
        return [tokenize_unlabeled(tweet, self.tokenizer, self.device, max_length=self.max_length) for tweet in tweets]

    def forward(self, x: Tensor):
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict_scores(self, tweets: List[Tweet]) -> List[List[float]]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        return self.forward(tensorized).sigmoid().cpu()

    def predict(self, tweets: List[Tweet]) -> List[str]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        preds = self.forward(tensorized).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def collate_tuples(pairs: List[Tuple[Tensor, LongTensor]], padding_value: int, device: str = 'cpu') -> Tuple[Tensor, LongTensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value).to(device), stack(ys).to(device)


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def tokenize_text(text: str, tokenizer: AutoTokenizer, device: str, **kwargs) -> Tensor:
    return tensor(tokenizer.encode(text, truncation=True, **kwargs), dtype=longt, device=device)


def tokenize_labels(labels: List[Label], device: str) -> LongTensor:
    return tensor([0 if label is False or label is None else 1 for label in labels], dtype=longt, device=device)


def tokenize_unlabeled(tweet: Tweet, tokenizer: AutoTokenizer, device: str, **kwargs) -> Tensor:
    return tokenize_text(tweet.text, tokenizer, device, **kwargs)


def tokenize_labeled(tweet: LabeledTweet, tokenizer: AutoTokenizer, device: str, **kwargs) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer, device, **kwargs), tokenize_labels(tweet.labels, device)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer, device: str, **kwargs) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer, device, **kwargs) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, device: str, **kwargs) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer, device, **kwargs) for tweet in read_unlabeled(path)]


def make_model(name: str, device: str = 'cpu') -> BERTLike:
    # todo: find all applicable models
    if name == 'del-covid':
        return BERTLike('digitalepidemiologylab/covid-twitter-bert-v2', device, model_dim=1024)
    elif name == 'vinai-covid':
        return BERTLike('vinai/bertweet-covid19-base-cased', device, max_length=128)
    elif name == 'vinai-tweet':
        return BERTLike('vinai/bertweet-base', device, max_length=128)
    elif name == 'cardiffnlp-tweet':
        return BERTLike('cardiffnlp/twitter-roberta-base', device)
    elif name == 'cardiffnlp-sentiment':
        return BERTLike('cardiffnlp/twitter-roberta-base-sentiment', device)
    elif name == 'cardiffnlp-hate':
        return BERTLike('cardiffnlp/twitter-roberta-base-hate', device)
    elif name == 'cardiffnlp-emotion':
        return BERTLike('cardiffnlp/twitter-roberta-base-emotion', device)
    elif name == 'cardiffnlp-offensive':
        return BERTLike('cardiffnlp/twitter-roberta-base-offensive', device)
    elif name == 'cardiffnlp-irony':
        return BERTLike('cardiffnlp/twitter-roberta-base-irony', device)
    
    # multi-lingual models
    elif name == 'multi-bert':
        return BERTLike('bert-base-multilingual-cased', device)
    elif name == 'multi-xlm':
        return BERTLike('xlm-roberta-base', device)
    elif name == 'multi-xnli':
        return BERTLike('joeddav/xlm-roberta-large-xnli', device, model_dim=1024)
    elif name == 'multi-microsoft':
        return BERTLike('microsoft/Multilingual-MiniLM-L12-H384', device, model_dim=384, token_name='xlm-roberta-base')
    elif name == 'multi-sentiment':
        return BERTLike('socialmediaie/TRAC2020_ALL_C_bert-base-multilingual-uncased', device)
    elif name == 'multi-toxic':
        return BERTLike('unitary/multilingual-toxic-xlm-roberta', device)

    else:
        raise ValueError(f'unknown name {name}')