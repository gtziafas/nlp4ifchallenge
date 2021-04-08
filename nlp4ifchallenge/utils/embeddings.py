from ..types import *
import torch

WordEmbedder = Callable[[List[str]], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sent: List[str]) -> array:
        sent_proc = _glove(sent)
        return array([word.vector for word in sent_proc])
    def embedd_many(tweets: List[LabeledTweet]) -> List[array]:
        return list(map(embedd, [t.text for t in tweets]))
    return embedd_many


# last hidden layer representations of a pre-trained BERT encoder
@torch.no_grad()
def frozen_bert_embeddings(name: str, from_checkpoint: Maybe[str] = None, **kwargs) -> WordEmbedder:
    from nlp4ifchallenge.models.bert import make_model
    model = make_model(name=name, **kwargs)
    if from_checkpoint is not None:
        model.load_state_dict(torch.load(from_checkpoint)['model_state_dict'])
    def embedd_many(sents: List[LabeledTweet]) -> array:
        return model.last_hidden_state(sents).cpu().numpy()
    return embedd_many


# make word embedder function
def make_word_embedder(embeddings: str, **kwargs) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)

    elif 'bert' in embeddings:
        embedder = frozen_bert_embeddings(name=embeddings, **kwargs)


    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')

    return embedder


