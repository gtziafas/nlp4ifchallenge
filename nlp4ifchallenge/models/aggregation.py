from ..types import *
from ..utils.metrics import preds_to_str

from torch import tensor, stack, no_grad, ones_like, zeros_like, where
from torch.nn import Module, Linear, ModuleList, Dropout, Conv1d


def aggregate_scores(scores: List[array], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths)   # (7,)
    predictions = (sum([score * faith for score, faith in zip(scores, faiths)])/sum_faiths).round().astype(int).tolist()
    return [preds_to_str(p) for p in predictions]


def aggregate_votes(scores: Tensor) -> List[str]:
    assert len(scores.shape) == 3, 'Must give B x M x Q float tensor'
    ones_per_q = scores.round().sum(dim=1)
    zeros_per_q = scores.shape[1] - ones_per_q
    votes = where(ones_per_q >= zeros_per_q, ones_like(ones_per_q), zeros_like(ones_per_q)).long() # B x Q
    return [preds_to_str(p) for p in votes.tolist()]


class PerQMetaClassifier(Module):
    def __init__(self, num_models: int, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.num_models = num_models
        self.dropout = Dropout(p=dropout)
        self.fc1 = Linear(in_features=num_models, out_features=hidden_size)
        self.fc2 = Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x).tanh() # B x M -> B x H
        x = self.dropout(x)
        return self.fc2(x) #  B x 1


class MetaClassifier(Module):
    def __init__(self, num_models: int, hidden_size: int, dropout: float, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        #self.perq_cls = ModuleList([PerQMetaClassifier(num_models, hidden_size, dropout) for _ in range(num_classes)])
        self.perq_cls = ModuleList([Linear(num_models, 1) for _ in range(num_classes)])

    def forward(self, inputs: Tensor) -> Tensor:
        xs = [x.squeeze(-1) for x in inputs.chunk(self.num_classes, dim=-1)] # [B x M]
        return stack([_cls.forward(x) for _cls, x in zip(self.perq_cls, xs)], dim=-1).squeeze(1)

    @no_grad()
    def predict(self, scores: Tensor, thresholds: Maybe[Tensor] = None) -> List[str]:
        self.eval()
        thresholds = thresholds if thresholds is not None else tensor([0.5] * self.num_classes, device=scores.device)
        aggr = self.forward(scores).sigmoid().ge(thresholds).long()  # B x Q
        return [preds_to_str(p) for p in aggr.cpu().tolist()]


class MetaClassifier2(Module):
    def __init__(self, num_models: int, hidden_size: int, dropout: float, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.fc = Linear(in_features=num_models * num_classes, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1) # B x MQ
        return self.fc(x) # B x Q