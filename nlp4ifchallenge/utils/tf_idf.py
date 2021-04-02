from ..types import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion

TfIdfTransform = Pipeline


def extract_tf_idfs(text: List[str],
        word_ngram_range: Tuple[int, int] = (1, 3),
        char_ngram_range: Tuple[int, int] = (3, 9),
        lsa_components: int = 100):
    print(f'{lsa_components} LSA components.')

    # extract tf-idf ngram features for each sentence in both word and char lvl    
    features = FeatureUnion([('tf_idf_word', TfidfVectorizer(ngram_range=word_ngram_range)),
                             ('tf_idf_char', TfidfVectorizer(analyzer='char', ngram_range=char_ngram_range))
                            ])

    # apply Latent Semantic Analysis (LSA) as truncated Singular Value Decomposition (SVD)
    # to reconstruct sparse tf-idf matrices in low-dimensions
    lsa = TruncatedSVD(n_components=lsa_components, random_state=0)

    # fit transformation pipeline into text
    tf = Pipeline([('feature_extraction', features),
                   ('latent_semantic_analysis', lsa)]).fit(text)

    # return embeds but also the transform to use in test-time
    return tf.transform(text).astype('float32'), tf