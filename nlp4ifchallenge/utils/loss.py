from ..types import *
from torch.nn import Module, BCEWithLogitsLoss


class BCELogitsIgnore(Module):
    def __init__(self, ignore_index: int, **kwargs):
        super(BCELogitsIgnore, self).__init__()
        self.ignore_index = ignore_index
        self.core = BCEWithLogitsLoss(**kwargs)

    def forward(self, inputs: Tensor, target: Tensor):
        inputs = inputs[target.ne(self.ignore_index)]
        target = target[target.ne(self.ignore_index)]
        return self.core(inputs, target)
