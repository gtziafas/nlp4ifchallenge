from ..types import *
from ..utils.metrics import preds_to_str


def aggregate_scores(scores: List[array], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths)   # (7,)
    predictions = (sum([score * faith for score, faith in zip(scores, faiths)])/sum_faiths).round().astype(int).tolist()
    return [preds_to_str(p) for p in predictions]


def aggregate_votes(votes: List[List[int]], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths) # (7,)
    predictions = [model.predict(tweets) for model in models]