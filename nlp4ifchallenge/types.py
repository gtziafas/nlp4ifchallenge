from typing import List, Tuple, Callable, TypeVar, Any, overload, Dict
from typing import Optional as Maybe
from dataclasses import dataclass
from abc import ABC

from torch import Tensor, LongTensor, tensor
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np 

from torch.nn import Module, Linear, BCEWithLogitsLoss, Dropout

array = np.array 

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
    faith: array

    def predict(self, tweets: List[Tweet]) -> List[str]:
        ...

    def predict_scores(self, tweets: List[Tweet]) -> array:
        ...
