from ...types import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def extract_tf_idfs(text: List[str], lsa_components: int=100, ngram_range_top: int=8) -> Tuple[array, TruncatedSVD, TfidfVectorizer]:

    print(f'{lsa_components} LSA components.')
    # extract tf-idf features for each sentence in char lvl
    tf_idf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, ngram_range_top))

    tf_idf = tf_idf_vectorizer.fit_transform(text)

    # apply Latent Semantic Analysis (LSA) as truncated Singular Value Decomposition (SVD)
    # to reconstruct sparse tf-idf matrices in low-dimensions
    lsa = TruncatedSVD(n_components=lsa_components, random_state=42).fit(tf_idf)

    # return embeds but also transforms to use in test-time
    return lsa.transform(tf_idf).astype('float32'), lsa, tf_idf_vectorizer

