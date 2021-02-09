"""LSTM model set-up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import progressbar
import time

def tokens_to_matrix(tokens, dictionary_size):
    """Helper function for converting a list of integers to a matrix with [0, 0, ..., 0, 1, 0, 0].

    Args:
        tokens: tokens to convert.
        dictionary_size: the size of the dictionary.

    Returns:
        np array with the result.
    """
    result = np.zeros((len(tokens), dictionary_size))
    for i, token in enumerate(tokens):
        result[i][token] = 1
    return result

def train(model, train_tokens, valid_tokens=None, test_tokens=None, logger=None,
          number_of_epochs=1):
    """Train the model in the train data.

    Args:
        model: the model to train.
        train_tokens: integer tokens for training.
        valid_tokens: integer tokens for validation.
        test_tokens: integer tokens for testing.
        logger: the logger to use.
        number_of_epochs: the number of epochs to run.
    """
    train_data = tokens_to_matrix(train_tokens, model.dictionary_size)
    valid_data = tokens_to_matrix(valid_tokens, model.dictionary_size) if valid_tokens else None
    test_data = tokens_to_matrix(test_tokens, model.dictionary_size) if test_tokens else None
    with progressbar.ProgressBar(max_value=number_of_epochs) as progress_bar:
        for epoch in range(number_of_epochs):
            # TODO(someone): implement training here.
            time.sleep(1)
            progress_bar.update(epoch + 1)


class Embedding(nn.Module):
    """The word to vector transformation.
    """
    def __init__(self, dictionary_size, embedding_size):
        """Initialization for embedding.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
        """
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.linear = nn.Parameter(torch.Tensor(dictionary_size, embedding_size))

    def forward(self, X):
        return self.linear(X)

class Model(nn.Module):
    def __init__(self, dictionary_size, embedding_size=10, number_of_layers=1,
                 droupout_probability=0.1):
        """Initialization for the model.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers.
            droupout_probability: the probability for dropping individual node in the network.
        """
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.number_of_layers = number_of_layers
        self.embedding = Embedding(dictionary_size, embedding_size)
        rnns = [nn.LSTM(embedding_size, embedding_size) for _ in range(number_of_layers)]
        self.rnns = nn.ModuleList(rnns)
        self.fc = nn.Linear(embedding_size, dictionary_size)
        self.dropout = nn.Dropout(p=droupout_probability)

    def forward(self, X, states):
        X = self.embed(X)
        X = self.dropout(X)
        for i, rnn in enumerate(self.rnns):
            X, states[i] = rnn(X, states[i])
            X = self.dropout(X)
        output = self.fc(X)
        return output, states
