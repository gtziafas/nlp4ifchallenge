from ..types import *
from ..utils.metrics import preds_to_str

from torch import tensor, stack, no_grad
from torch.nn import Module, Linear, ModuleList


def aggregate_scores(scores: List[array], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths)   # (7,)
    predictions = (sum([score * faith for score, faith in zip(scores, faiths)])/sum_faiths).round().astype(int).tolist()
    return [preds_to_str(p) for p in predictions]


class PerQMetaClassifier(Module):
    def __init__(self, num_models: int, hidden_size: int):
        super().__init__()
        self.num_models = num_models
        self.fc1 = Linear(in_features=num_models, out_features=hidden_size)
        self.fc2 = Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        # B x M
        x = self.fc1(x).tanh() # B x H
        return self.fc2(x) #  7 x 1


class MetaClassifier(Module):
    def __init__(self, num_models: int, hidden_size: int, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.perq_cls = ModuleList([PerQMetaClassifier(num_models, hidden_size) for _ in range(num_classes)])

    def forward(self, inputs: Tensor) -> Tensor:
        # B x M x Q
        xs = [x.squeeze(-1) for x in inputs.chunk(self.num_classes, dim=-1)] # [B X M, ...]
        return stack([_cls.forward(x) for _cls, x in zip(self.perq_cls, xs)], dim=-1).squeeze(1) # B X 

    @no_grad()    
    def aggregate(self, scores: Tensor, thresholds: Maybe[Tensor] = None) -> List[str]:
        self.eval()
        thresholds = thresholds if thresholds is not None else tensor([0.5] * self.num_classes, device=scores.device)
        aggr = self.forward(scores) # B x Q
        return [preds_to_str(p) for p in (aggr > thresholds).long().cpu().tolist()]