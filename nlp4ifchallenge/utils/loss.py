from ..types import *
from torch import Module, exp
import torch.nn.functional as F


class WeightedFocalLoss(Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, device: str):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.long()
        at = self.alpha.gather(0, targets.view(-1))
        pt = exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()