from ..types import * 
from numpy import argwhere
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


def _round(x: float) -> float:
    return round(x, 3)


def preds_to_str(preds: List[int]) -> str:
    return '\t'.join(['nan' if preds[0] == 0 and 0 < i < 5 else 'yes' if p == 1 else 'no' for i, p in enumerate(preds)])


def get_metrics(preds: List[List[int]], labels: List[List[int]], ignore_index: int = -1) -> Dict[str, Any]:
    preds_ = [preds_to_str(p).split('\t') for p in preds]
    labels_ = [preds_to_str(l).split('\t') for l in labels]
    per_column_preds = list(zip(*preds_))
    per_column_labels = list(zip(*labels_))
    per_column_metrics = [(f1_score(pcp, pcl, labels=['yes', 'no'], average='weighted'),
                           recall_score(pcp, pcl, labels=['yes', 'no'], average='weighted'),
                           precision_score(pcp, pcl, labels=['yes', 'no'], average='weighted'))
                          for pcp, pcl in zip(per_column_preds, per_column_labels)]
    f1s, ps, rs = list(zip(*per_column_metrics))

    # manually remove ignored nans for accuracy and hamming computation 
    preds, labels = array(preds), array(labels)
    labels[labels == ignore_index] = 0
    preds[argwhere(preds[:,0] == 0), 1:5] = 0
    return {'accuracy': _round(accuracy_score(preds, labels)),
            'hamming': _round(1 - hamming_loss(preds, labels)),
            'mean_f1': _round(sum(f1s)/len(f1s)),
            'mean_precision': _round(sum(ps)/len(ps)),
            'mean_recall': _round(sum(rs)/len(rs)),
            'column_wise':
                [{'f1': _round(f1), 'precision': _round(p), 'recall': _round(r)} for f1, p, r in per_column_metrics]
            }
