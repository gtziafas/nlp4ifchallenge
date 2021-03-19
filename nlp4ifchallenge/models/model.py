from ..types import *


class Model(Protocol):
    def predict(self, tweets: List[Tweet]) -> List[List[bool]]:
        ...
