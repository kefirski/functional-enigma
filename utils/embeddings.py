import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class EmbedFetcher:
    @staticmethod
    def fetch(words, path, binary=True):

        embeddings = KeyedVectors.load_word2vec_format(path, binary=binary)

        def hook(word):
            if word in embeddings.vocab:
                vector = embeddings.word_vec(word)
                return vector / np.linalg.norm(vector, ord=2)
            else:
                return np.zeros(embeddings.vector_size)

        return np.stack([hook(word) for word in words])