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
    def embedd_many(sents: List[List[str]]) -> List[array]:
        return list(map(embedd, sents))
    return embedd_many


# last hidden layer representations of a pre-trained BERT encoder
@torch.no_grad()
def frozen_bert_embeddings(name: str, **kwargs) -> WordEmbedder:
    from ..models.bert import BertLike
    model = BertLike(name=name, **kwargs)
    def embedd_many(sents: List[List[str]]) -> array:
        tokens = stack(model.tensorize_labeled(sents))
        attention_mask = tokens.ne(model.tokenizer.pad_token_id)
        hidden, _ = model(tokens, attention_mask, output_hidden_states=True, return_dict=False)
        return hidden.cpu().numpy()
    return embedd_many


# make word embedder function
def make_word_embedder(embeddings: str) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)

    elif 'bert' in embeddings:
        embedder = frozen_bert_embeddings(name=embeddings)


    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')

    return embedder


