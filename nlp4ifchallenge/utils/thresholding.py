from ..types import * 
from ..utils.metrics import f1_score


def find_thresholds(logits: Tensor, labels: List[List[int]], repeats: int):
    per_q_logits = [pql.squeeze(-1).tolist() for pql in logits.chunk(7, dim=-1)]
    per_q_labels = list(zip(*labels))
    for i, (pql, pqt) in enumerate(zip(per_q_logits, per_q_labels)):
        print('=' * 64)
        print(i)
        print('=' * 64)
        predictions, truths = list(zip(*[(p, t) for p, t in zip(pql, pqt) if t != -1]))
        min_t, cur_t, max_t = (0.35, 0.5, 0.65)
        for repeat in range(repeats):
            thresholds = [min_t, cur_t, max_t]
            f1s = [get_f1_at_threshold(predictions, truths, threshold) for threshold in thresholds]
            print(list(zip(thresholds, f1s)))
            low, high = (max(left, right) for left, right in zip(f1s, f1s[1:]))
            if low > high:
                max_t = cur_t
            elif low == high:
                min_t = (cur_t - min_t) / 2 + min_t
                max_t = (max_t - cur_t) / 2 + cur_t
            else:
                min_t = cur_t
            cur_t = min_t + (max_t - min_t) / 2


def get_f1_at_threshold(predictions: List[float], truths: List[int], threshold: float) -> float:
    rounded = [1 if p >= threshold else 0 for p in predictions]
    return f1_score(truths, rounded, average='weighted', labels=[1, 0])