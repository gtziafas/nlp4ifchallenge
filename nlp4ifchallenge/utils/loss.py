from ..types import *
from torch import Module, BCEWithLogitsLoss, exp


class WeightedFocalLoss(Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, device: str = 'cpu', class_weights: Maybe[Tensor] = None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma
        self.bce = BCEWithLogitsLoss(reduction='none', pos_weights=class_weights)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        BCE_loss = self.bce(inputs, targets) 
        targets = targets.long()
        at = self.alpha.gather(0, targets.view(-1))
        pt = exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()