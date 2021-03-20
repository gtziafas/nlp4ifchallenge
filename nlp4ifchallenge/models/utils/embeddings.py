from ...types import *

WordEmbedder = Callable[[Tweet], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sent: Tweet) -> array:
        sent_proc = _glove(sent)
        return array([word.vector for word in sent_proc])
    return embedd


# make word embedder function
def make_word_embedder(embeddings: str) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)

    # elif embeddings == 'bert':
    #     embedder = bert_pretrained_embeddings()


    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')

    return embedder

# aggregates features from all word vectors in paragraph by average pooling
def bag_of_embeddings(sents: List[array]) -> array:
    return array([sent.mean(axis=0) for sent in sents])

