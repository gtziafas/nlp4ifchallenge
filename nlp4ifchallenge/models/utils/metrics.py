from ...types import * 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from ..bert import preds_to_str


def _round(x: float) -> float:
    return round(x, 3)


def get_metrics(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    preds_ = [preds_to_str(p).split('\t') for p in preds]
    labels_ = [preds_to_str(l).split('\t') for l in labels]
    per_column_preds = list(zip(*preds_))
    per_column_labels = list(zip(*labels_))
    per_column_metrics = [(f1_score(pcp, pcl, labels=['yes', 'no'], average='weighted'),
                           recall_score(pcp, pcl, labels=['yes', 'no'], average='weighted'),
                           precision_score(pcp, pcl, labels=['yes', 'no'], average='weighted'))
                          for pcp, pcl in zip(per_column_preds, per_column_labels)]
    f1s, ps, rs = list(zip(*per_column_metrics))
    return {'accuracy': _round(accuracy_score(preds, labels)),
            'hamming': _round(1 - hamming_loss(preds, labels)),
            'mean_f1': _round(sum(f1s)/len(f1s)),
            'mean_p': _round(sum(ps)/len(ps)),
            'mean_r': _round(sum(rs)/len(rs)),
            'column_wise':
                [{'f1': _round(f1), 'p': _round(p), 'r': _round(r)} for f1, p, r in per_column_metrics]
            }
