from typing import List, Tuple, Callable, TypeVar, Any, overload, Literal, Protocol
from typing import Optional as Maybe
from dataclasses import dataclass

from torch import Tensor

Label = Maybe[bool]

T1 = TypeVar('T1')


@dataclass
class Tweet:
    no:     int
    text:   str


@dataclass
class LabeledTweet(Tweet):
    labels: [Label] * 7
