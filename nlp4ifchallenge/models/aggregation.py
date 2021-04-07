from ..types import *
from ..utils.metrics import preds_to_str

from torch import tensor, stack, no_grad
from torch.nn import Module, Linear, ModuleList, Dropout, Conv1d


def aggregate_scores(scores: List[array], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths)   # (7,)
    predictions = (sum([score * faith for score, faith in zip(scores, faiths)])/sum_faiths).round().astype(int).tolist()
    return [preds_to_str(p) for p in predictions]


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
        self.perq_cls = ModuleList([PerQMetaClassifier(num_models, hidden_size, dropout) for _ in range(num_classes)])
        #self.perq_cls = ModuleList([Linear(num_models, 1) for _ in range(num_classes)])

    def forward(self, inputs: Tensor) -> Tensor:
        xs = [x.squeeze(-1) for x in inputs.chunk(self.num_classes, dim=-1)] # [B x M]
        return stack([_cls.forward(x) for _cls, x in zip(self.perq_cls, xs)], dim=-1).squeeze(1)

    @no_grad()
    def threshold(self, scores: Tensor, thresholds: Maybe[Tensor] = None) -> List[str]:
        self.eval()
        thresholds = thresholds if thresholds is not None else tensor([0.5] * self.num_classes, device=scores.device)
        aggr = self.forward(scores)  # B x Q
        return [preds_to_str(p) for p in (aggr.ge(thresholds)).long().cpu().tolist()]


class MetaClassifier2(Module):
    def __init__(self, num_models: int, hidden_size: int, dropout: float, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = Dropout(p=dropout)
        self.gelu = GELU()
        self.conv1 = Conv1d(in_channels=num_classes, out_channels=hidden_size, kernel_size=num_models//2 + 1, stride=1)
        self.conv2 = Conv1d(in_channels=hidden_size, out_channels=num_classes, kernel_size=num_models//2, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-1, 1) # B x M x Q -> B x Q x M 
        x = self.conv1(x)  # B x H x M/2
        x = self.dropout(x.tanh()) 
        x = self.conv2(x)  # B x Q x 1
        return x.squeeze(-1) # B x Q

    @no_grad()
    def threshold(self, scores: Tensor, thresholds: Maybe[Tensor] = None) -> List[str]:
        self.eval()
        thresholds = thresholds if thresholds is not None else tensor([0.5] * self.num_classes, device=scores.device)
        aggr = self.forward(scores)  # B x Q
        return [preds_to_str(p) for p in (aggr.ge(thresholds)).long().cpu().tolist()]
