from ..types import *
from .utils.metrics import preds_to_str
from numpy import array


def aggregate_predictions(tweets: List[Tweet], models: List[Model], faiths: List[array]) -> List[str]:
    sum_faiths = sum(faiths)   # (7,)
    predictions = (sum([model.predict_scores(tweets) * faith
                        for model, faith in zip(models, faiths)])/sum_faiths).round().astype(int).tolist()
    return [preds_to_str(p) for p in predictions]
