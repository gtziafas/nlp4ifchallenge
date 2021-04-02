from ..types import *
from ..utils.embeddings import WordEmbedder, make_word_embedder
from ..utils.tf_idf import extract_tf_idfs, TfIdfTransform

from torch.nn import Module, Linear, GRU, Dropout, GELU, Sequential
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import tensor, stack, cat

MaybeTensorPair = Tuple[Tensor, Maybe[Tensor]]


# bagging model aggregating word-embedds and classifying with an MLP head
class BaggingModel(Module):
    def __init__(self, aggregator: Module, classifier: Module):
        super().__init__()
        self.aggregator = aggregator
        self.cls = classifier

    # optionally concat tf_idf reps to the pooled sent vectors
    def pool(self, x: Tensor, t: Maybe[Tensor]) -> Tensor:
        pooler = self.aggregator(x)
        if t is not None:
            pooler = cat((pooler, t), dim=-1)
        return pooler

    def forward(self, inputs: MaybeTensorPair) -> Tensor:
        x, t = inputs
        features = self.pool(x, t)
        return self.cls(features)
        

# a wrapper for using a bagging model in test time
class BaggingModelTest(BaggingModel, Model):
    def __init__(self, aggregator: Module, classifier: Module, word_embedder: WordEmbedder, 
            device: str, tf_idf_transform: Maybe[TfIdfTransform] = None):
        super().__init__(aggregator, classifier)
        self.we = word_embedder 
        self.tf_idf = tf_idf_transform
        self.device = device

    def embedd(self, sents: List[Sentence]) -> Tuple[Tensor, Maybe[Tensor]]:
        text = [sent.text for sent in sents] 
        word_embedds = tensor(array(self.we(text)), dtype=floatt, device=self.device)
        tf_idfs = None
        if self.tf_idf is not None:
            tf_idfs = tensor(self.tf_idf.transform(text), dtype=floatt, device=self.device)
        return word_embedds, tf_idfs

    def predict(self, sents: List[Sentence]) -> List[int]:
        inputs = self.embedd(sents)
        return self.forward(inputs).sigmoid().round().long().cpu().tolist()

    def predict_scores(self, sents: List[Sentence]) -> array:
        inputs = self.embedd(sents)
        return self.forward(inputs).sigmoid().cpu().tolist()
        

# bag of embedds aggregation with average pooling
class BagOfEmbeddings(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1).squeeze() # B, S, D -> B, D


# bi-GRU contextualization as aggregator
class RNNContext(Module):
    def __init__(self, inp_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = GRU(input_size=inp_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        dh = self.rnn.hidden_size
        h, _ = self.rnn(x)
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


def tensorize_unlabeled(sents: List[Sentence], we: WordEmbedder, tf_idf: Maybe[TfIdfTransform], device: str = 'cpu') -> List[MaybeTensorPair]:
    text = [sent.text for sent in sents]
    word_embedds = [tensor(sent, dtype=floatt, device=device) for sent in we(text)]
    tfs = [None] * len(text)
    if tf_idf is not None:
        tfs = tf_idf.transform(text)
        tfs = [tensor(t, dtype=floatt, device=device) for t in tfs.tolist()]
    return list(zip(word_embedds, tfs))


def tensorize_labeled(sents: List[AnnotatedSentence], *args, **kwargs) -> List[Tuple[Tensor, Maybe[Tensor], Tensor]]:
    unlabeled = tensorize_unlabeled([Sentence(sent.no, sent.text) for sent in sents], *args, **kwargs)
    labels = tokenize_labels([sent.labels for sent in sents])
    return [(inp[0], inp[1], label) for inp, label in zip(unlabeled, labels)]


def tokenize_labels(labels: List[List[Label]], device: str = 'cpu') -> List[Tensor]:
    def _tokenize_labels(labels: List[Label]) -> Tensor:
        return tensor([0 if label is False else 1 for label in labels], dtype=longt, device=device)
    return list(map(_tokenize_labels, labels))


def collate_tuples(tuples: List[Tuple[Tensor, Maybe[Tensor], Tensor]], padding_value: int = 0) -> Tuple[MaybeTensorPair, Tensor]:
    xs, ts, ys = zip(*tuples)
    xs = pad_sequence(xs, padding_value)
    ts = None if ts[0] is None else stack(ts, dim=0).float()
    ys = stack(ys, dim=0)
    return (xs, ts), ys


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def make_model(kwargs: Dict) -> BaggingModel:
    if kwargs['aggregator'] == 'BoE':
        _agg = BagOfEmbeddings()
    
    elif kwargs['aggregator'] == 'RNN':
        inp_dim = kwargs['inp_dim']
        _agg = RNNContext(inp_dim, inp_dim // 2)

    else:
        raise ValueError(f'unknown aggregator method {kwargs["aggregator"]}')

    if not kwargs['hidden_dim']:
        _cls = Linear(kwwargs['inp_dim'], kwargs['num_classes']) 
    else: 
        _cls = MLPHead(kwargs['inp_dim'] + kwargs['with_tf_idf'], kwargs['hidden_dim'], kwargs['num_classes'], kwargs['dropout'])

    return BaggingModel(aggregator=_agg, classifier=_cls)
