from ...types import * 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
import torch


def get_metrics(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    return {'accuracy': accuracy_score(preds, labels),
            'hamming': 1 - hamming_loss(preds, labels),
            'f1': f1_score(preds, labels, average='weighted'),
            'r': recall_score(preds, labels, average='weighted'),
            'p': precision_score(preds, labels, average='weighted')}
