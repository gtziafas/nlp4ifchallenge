from typing import List, Tuple, Callable, TypeVar, Any, overload
from typing import Optional as Maybe
from dataclasses import dataclass
from abc import ABC

from torch import Tensor

from torch.nn import Module

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
    def predict(self, tweets: List[Tweet]) -> List[str]:
        ...
