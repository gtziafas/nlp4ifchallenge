from ...types import *
from ..bert import *
import torch

WordEmbedder = Callable[[Tweet], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sent: Tweet) -> array:
        sent_proc = _glove(sent)
        return array([word.vector for word in sent_proc])
    return embedd


# last hidden layer representations of a pre-trained BERT encoder
@torch.no_grad()
def frozen_bert_embeddings(name: str, **kwargs) -> WordEmbedder:
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    model = AutoModel.from_pretrained(name)
    def embedd(sent: Tweet) -> array:
        tokens = tokenize_text(sent, tokenizer, **kwargs).unsqueeze(0)
        attention_mask = tokens.ne(tokenizer.pad_token_id)
        hidden, _ = model(tokens, attention_mask, output_hidden_states=True, return_dict=False).squeeze()
        print(hidden.shape)
        return hidden.cpu().numpy()
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


