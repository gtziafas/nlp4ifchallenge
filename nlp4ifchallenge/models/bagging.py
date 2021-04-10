from ..types import *
from ..utils.embeddings import WordEmbedder, make_word_embedder
from ..utils.tf_idf import extract_tf_idfs, TfIdfTransform
from ..utils.metrics import preds_to_str

from torch.nn import Module, Linear, GRU, Dropout, GELU, Sequential
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import tensor, stack, cat, tanh

MaybeTensorPair = Tuple[Tensor, Maybe[Tensor]]


# bagging model pooling word-embedds and classifying with an MLP head
class BaggingModel(Module):
    def __init__(self, pooler: Module, classifier: Module):
        super().__init__()
        self.pooler = pooler
        self.cls = classifier

    # optionally concat tf_idf reps to the pooled sent vectors
    def pool(self, x: Tensor, t: Maybe[Tensor]) -> Tensor:
        pooler = self.pooler(x)
        if t is not None:
            gating = tanh if pooler.min().item() < 0 else lambda x: x
            pooler = cat((pooler, gating(t)), dim=-1)
        return pooler

    def forward(self, inputs: MaybeTensorPair) -> Tensor:
        x, t = inputs
        features = self.pool(x, t)
        return self.cls(features)
        

# a wrapper for using a bagging model in test time
class BaggingModelTest(BaggingModel, Model):
    def __init__(self, pooler: Module, classifier: Module, word_embedder: WordEmbedder, 
            device: str, tf_idf_transform: Maybe[TfIdfTransform] = None):
        super().__init__(pooler, classifier)
        self.we = word_embedder 
        self.tf_idf_transform = tf_idf_transform
        self.device = device

    def embedd(self, tweets: List[Tweet]) -> MaybeTensorPair:
        text = [tweet.text for tweet in tweets] 
        word_embedds = tensor(array(self.we(text)), dtype=floatt, device=self.device)
        tf_idfs = None
        if self.tf_idf_transform is not None:
            tf_idfs = tensor(self.tf_idf_transform.transform(text), dtype=floatt, device=self.device)
        return word_embedds, tf_idfs

    def predict(self, tweets: List[Tweet]) -> List[str]:
        inputs = self.embedd(tweets)
        preds = self.forward(inputs).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]

    def predict_scores(self, tweets: List[Tweet], threshold: float = 0.5) -> array:
        inputs = self.embedd(tweets)
        return self.forward(inputs).ge(threshold).cpu().tolist()
        

# bag of embedds representation with average pooling
class BagOfEmbeddings(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1) # B x S x D -> B x D


# bi-GRU contextualization as pooler
class GRUContext(Module):
    def __init__(self, inp_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = GRU(input_size=inp_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        dh = self.gru.hidden_size
        h, _ = self.gru(x)
        context = cat((h[:, -1, :dh], h[:, 0, dh:]), dim=-1)
        return context 


# 2-layer MLP head for classification
class MLPHead(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.33):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.hidden = Linear(in_features=inp_dim, out_features=hidden_dim)
        self.out = Linear(in_features=hidden_dim, out_features=num_classes)
        self.gelu = GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.out(x)


def tensorize_unlabeled(tweets: List[Tweet], we: WordEmbedder, tf_idf_transform: Maybe[TfIdfTransform], device: str = 'cpu') -> List[MaybeTensorPair]:
    text = [tweet.text for tweet in tweets]
    word_embedds = we(text)
    word_embedds = [tensor(emb, dtype=floatt, device=device) for emb in word_embedds]
    tf_idfs = [None] * len(text)
    if tf_idf_transform is not None:
        tf_idfs = tf_idf_transform.transform(text)
        tf_idfs = [tensor(t, dtype=floatt, device=device) for t in tf_idfs.tolist()]
    return list(zip(word_embedds, tf_idfs))


def tensorize_labeled(tweets: List[LabeledTweet], *args, **kwargs) -> List[Tuple[Tensor, Maybe[Tensor], LongTensor]]:
    unlabeled = tensorize_unlabeled(tweets, *args, **kwargs)
    labels = tokenize_labels([tweet.labels for tweet in tweets])
    return [(inp[0], inp[1], label) for inp, label in zip(unlabeled, labels)]


def tokenize_labels(labels: List[List[Label]], ignore_nan: bool, device: str = 'cpu') -> List[LongTensor]:
    nan_label = 0 if not ignore_nan else -1
    def _tokenize_labels(_labels: List[Label]) -> LongTensor:
        return tensor([nan_label if label is None else 0 if label is False else 1 for label in _labels], dtype=longt, device=device)
    return list(map(_tokenize_labels, labels))


def collate_tuples(tuples: List[Tuple[Tensor, Maybe[Tensor], LongTensor]], padding_value: int = 0, device: str = 'cpu') -> Tuple[MaybeTensorPair, LongTensor]:
    xs, ts, ys = zip(*tuples)
    xs = pad_sequence(xs, padding_value).to(device)
    ts = None if ts[0] is None else stack(ts, dim=0).float().to(device)
    ys = stack(ys, dim=0).to(device)
    return (xs, ts), ys


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def make_model(kwargs: Dict) -> BaggingModel:
    if kwargs['pooler'] == 'BoE':
        inp_dim = kwargs['inp_dim']
        _agg = BagOfEmbeddings()
    
    elif kwargs['pooler'] == 'RNN':
        inp_dim = 2 * kwargs['hidden_dim_rnn']
        _agg = GRUContext(kwargs['inp_dim'], kwargs['hidden_dim_rnn'])

    else:
        raise ValueError(f'unknown pooler method {kwargs["pooler"]}')

    if not kwargs['hidden_dim_mlp']:
        _cls = Linear(inp_dim, kwargs['num_classes']) 
    else: 
        _cls = MLPHead(inp_dim + kwargs['with_tf_idf'], kwargs['hidden_dim_mlp'], kwargs['num_classes'], kwargs['dropout'])

    return BaggingModel(pooler=_agg, classifier=_cls)
