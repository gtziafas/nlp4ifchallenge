from ..types import *
from torch.nn import Module, BCEWithLogitsLoss


class BCEWithLogitsIgnore(Module):
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.core = BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions[targets.ne(self.ignore_index)]
        targets = targets[targets.ne(self.ignore_index)]
        return self.core(predictions, targets)