from typing import List, Tuple, Callable, TypeVar, Any, overload, Dict
from typing import Optional as Maybe
from dataclasses import dataclass
from abc import ABC

from torch import Tensor, LongTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import long as longt 
from torch import float as floatt
from numpy import array

Label = Maybe[bool]

T1 = TypeVar('T1')


@dataclass
class Tweet:
    no:     int
    text:   str


@dataclass
class LabeledTweet(Tweet):
    labels: [Label] * 7


class Model(ABC):

    def predict(self, tweets: List[Tweet], threshold: float) -> List[str]:
        ...

    def predict_scores(self, tweets: List[Tweet]) -> Tensor:
        ...
