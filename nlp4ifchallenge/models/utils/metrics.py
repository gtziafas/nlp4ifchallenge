from ...types import * 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


def get_metrics(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    # todo: check their shitty code
    return {'accuracy': round(accuracy_score(preds, labels), 3),
            'hamming': round(1 - hamming_loss(preds, labels), 3),
            'f1': round(f1_score(preds, labels, average='weighted'), 3),
            'r': round(recall_score(preds, labels, average='weighted'), 3),
            'p': round(precision_score(preds, labels, average='weighted'), 3),
            'per_column_f1': [round(pcf1, 3) for pcf1 in per_column_f1(preds, labels)]}


def per_column_f1(preds: List[List[int]], labels: List[List[int]]) -> List[float]:
    per_column_preds = list(zip(*preds))
    per_column_labels = list(zip(*labels))
    return [f1_score(pcp, pcl, average='weighted') for pcp, pcl in zip(per_column_preds, per_column_labels)]
