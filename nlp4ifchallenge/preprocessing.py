from .types import *


def read_labeled(file_path: str) -> List[LabeledTweet]:
    def parse_labeled(line: str) -> LabeledTweet:
        def parse_label(label: str) -> Label:
            return True if label == 'yes' else False if label == 'no' else None

        tabs = line.strip('\n').split('\t')
        nr, text, labels = tabs[0], tabs[1], tabs[2:]
        return LabeledTweet(int(nr), text, list(map(parse_label, labels)))
    with open(file_path, 'r') as f:
        _ = next(f)
        return [parse_labeled(line) for line in f]


def read_unlabeled(file_path: str) -> List[Tweet]:
    def parse_unlabeled(line: str) -> Tweet:
        tabs = line.split('\t')
        nr, text = tabs[0], tabs[1]
        return Tweet(int(nr), text)

    with open(file_path, 'r') as f:
        _ = next(f)
        return [parse_unlabeled(line) for line in f]


def tokenize_tweets(tweets: List[Tweet], tokenizer_fn: Callable[[Tweet], T1]) -> List[T1]:
    return [tokenizer_fn(tweet) for tweet in tweets]
