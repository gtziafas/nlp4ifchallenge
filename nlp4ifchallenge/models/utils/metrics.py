from ...types import * 

import torch

@torch.no_grad()
def multi_label_metrics(preds: Tensor, labels: LongTensor, thresh: float=0.5) -> Dict[str, float]:
    batch_size, num_classes = preds.shape

    # for evaluation, make all logit predictions >50% to become 1 and other 0
    preds = torch.where(preds.sigmoid()>thresh, torch.ones_like(preds), torch.zeros_like(preds))

    # accuracy score counts the percentage of correctly classified entire sentences
    corrects = preds == labels 
    count_mask = corrects.sum(dim=-1)
    accuracy = count_mask[count_mask==num_classes].shape[0] / batch_size

    # hamming loss counts the fraction of incorrectly classified specific labels
    hamm_loss = corrects[corrects==0].shape[0] / (batch_size * num_classes)

    return {'accuracy':accuracy, 'hamming_loss':hamm_loss} 
