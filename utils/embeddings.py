import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class EmbedFetcher:
    @staticmethod
    def fetch(words, path, binary=True):
        """
        :param words: An list of words to construct embedding matrix
        :param path: Path to embeddings to load from
        :return: ndarray with shape of [len(words), embed_size]
        """

        embeddings = KeyedVectors.load_word2vec_format(path, binary=binary)

        def hook(word):
            if word in embeddings.vocab:
                return embeddings.word_vec(word)
            else:
                return np.zeros(embeddings.vector_size)

        return np.stack([hook(word) for word in words])