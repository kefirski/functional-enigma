import collections
import os
import re
from copy import copy

import numpy as np
import pandas as pnd
import torch as t
from six.moves import cPickle
from torch.autograd import Variable

from .embeddings import EmbedFetcher


class Dataloader():
    def __init__(self, data_path='', embeddings_path='', force_preprocessing=False):
        """
        :param data_path: path to data
        :param force_preprocessing: whether to preprocess data even if it was preprocessed before
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path
        self.prep_path = self.data_path + 'preprocessings/'

        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)

        '''
        go_token (stop_token) uses to mark start (end) of the sequence
        pad_token uses to fill tensor to fixed-size length
        In order to make modules work correctly,
        these tokens should be unique
        '''
        self.go_token = '<GO>'
        self.pad_token = '<PAD>'
        self.stop_token = '<STOP>'

        self.pretrained_embeddings = embeddings_path

        self.data_file = self.data_path + 'short_text.csv'

        self.idx_file = self.prep_path + 'vocab.pkl'
        self.tensor_file = self.prep_path + 'tensor.pkl'
        self.preprocessed_embeddings = self.prep_path + 'embeddings.npy'

        idx_exists = os.path.exists(self.idx_file)
        tensor_exists = os.path.exists(self.tensor_file)
        embed_exists = os.path.exists(self.preprocessed_embeddings)

        preprocessings_exist = all([file for file in [idx_exists, tensor_exists, embed_exists]])

        if preprocessings_exist and not force_preprocessing:
            print('Loading preprocessed data have started')
            self.load_preprocessed()
            print('Preprocessed data have loaded')
        else:
            print('Processing have started')
            self.preprocess()
            print('Data have preprocessed')

    def build_vocab(self, sentences):
        """
        :param sentences: An array of chars in data
        :return:
            vocab_size – Number of unique words in corpus
            idx_to_word – Array of shape [vocab_size] containing list of unique chars
            word_to_idx – Dictionary of shape [vocab_size]
                such that idx_to_word[word_to_idx[some_char]] = some_char
                where some_char is is from idx_to_word
        """

        word_counts = collections.Counter(sentences)

        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = [self.pad_token, self.go_token, self.stop_token] + list(sorted(idx_to_word))

        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        vocab_size = len(idx_to_word)

        return vocab_size, idx_to_word, word_to_idx

    @staticmethod
    def clear_line(line):
        line = re.sub(r"[^a-z']", ' ', line)
        line = re.sub(r"'", " '", line)
        line = re.sub(r'\s{2,}', '', line)

        return line

    def preprocess(self):
        data = pnd.read_csv(self.data_file)[:10]

        data['text'] = data['text'].map(lambda line: self.clear_line(' '.join(str(line).lower().split(' ')[:100])))
        data['title'] = data['title'].map(lambda line: self.clear_line(' '.join(str(line).lower().split(' ')[:100])))

        words = [word for line in list(data['text']) for word in line.split(' ')] + \
                [word for line in list(data['title']) for word in line.split(' ')]
        self.vocab_size, self.idx_to_word, self.word_to_idx = self.build_vocab(words)

        embeddings = EmbedFetcher.fetch(self.idx_to_word, self.pretrained_embeddings)
        np.save(self.preprocessed_embeddings, embeddings)

        self.data = data
        self.data['text'] = data['text'] \
            .map(lambda line: [self.word_to_idx[self.go_token]] +
                              [self.word_to_idx[word] for word in line.split(' ')] +
                              [self.word_to_idx[self.stop_token]])
        self.data['title'] = data['title'] \
            .map(lambda line: [self.word_to_idx[self.go_token]] +
                              [self.word_to_idx[word] for word in line.split(' ')] +
                              [self.word_to_idx[self.stop_token]])

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):

        self.idx_to_word = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_word)
        self.word_to_idx = dict(zip(self.idx_to_word, range(self.vocab_size)))

        self.data = cPickle.load(open(self.tensor_file, "rb"))

    def next_batch(self, batch_size):
        """
        :param batch_size: number of selected data elements
        :return: target tensors
        """

        indexes = np.random.choice(list(self.data.index), size=batch_size)
        lines = self.data.ix[indexes]
        lines = [list(lines[target]) for target in ['text', 'title']]
        del indexes

        return self.construct_batches(lines)

    def construct_batches(self, lines):
        """
        :param lines: An list of indexes arrays
        :return: Batches
        """

        input = lines[0]
        decoder_input = [line[:-1] for line in lines[1]]
        target = [line[1:] for line in lines[1]]

        input = self.pad_input(input)
        decoder_input = self.pad_input(decoder_input)
        target = self.pad_input(target)

        return input, decoder_input, target

    @staticmethod
    def pad_input(sequences):

        lengths = [len(line) for line in sequences]
        max_length = max(lengths)

        '''
        Pad token has idx 0 for both targets
        '''
        return np.array([line + [0] * (max_length - lengths[i])
                         for i, line in enumerate(sequences)])

    def torch(self, batch_size, volatile=False):

        input, decoder_input, target = self.next_batch(batch_size)
        input, decoder_input, target = [Variable(t.from_numpy(var), volatile=volatile)
                                        for var in [input, decoder_input, target]]

        return input, decoder_input, target

    def go_input(self, batch_size):

        go_input = np.array([[self.word_to_idx[self.go_token]]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()

        return go_input

    def to_tensor(self, strings, volatile=True):
        tensor = Variable(t.LongTensor([[self.word_to_idx[self.go_token]] +
                                        [self.word_to_idx[word] for word in line.split(' ')[:-1]]
                                        for line in strings]), volatile=volatile)
        return tensor

    def sample_char(self, p, n_beams=5):

        p = [[i, val] for i, val in enumerate(p)]
        p = sorted(p, key=lambda pair: pair[1])[-n_beams:]
        return [(self.idx_to_word[idx], prob) for idx, prob in p]

    def beam_update(self, beams, probs):

        n_beams = len(probs)

        for i in range(len(beams)):
            probs[i] *= beams[i].prob

        probs = [[beam, idx, p] for i, beam in enumerate(beams) for idx, p in enumerate(probs[i])]
        probs = sorted(probs, key=lambda triple: triple[2])[-n_beams:]
        return [copy(beam).update(prob, self.idx_to_word[idx]) for beam, idx, prob in probs]
