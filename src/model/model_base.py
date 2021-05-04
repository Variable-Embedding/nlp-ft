"""Base Model Class Configuration.
"""
import torch
import torch.nn as nn

from src.model.model_tools import prep_embedding_layer
from src.model.model_lstm import LSTM
from src.model.model_ft import FT


class Model(nn.Module):
    def __init__(self
                 , dictionary_size
                 , embedding_size=100
                 , number_of_layers=1
                 , dropout_probability=0.3
                 , batch_size=64
                 , sequence_length=30
                 , max_norm=2
                 , max_init_param=0.01
                 , device="gpu"
                 , sequence_step_size=None
                 , lstm_configuration="default"
                 , embedding_vectors=None
                 , embedding_trainable=True
                 , model_type=None
                 , **kwargs
                 ):
        """Initialization for the model.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers.
            dropout_probability: the probability for dropping individual node in the network.
            batch_size: the batch size for the model.
            sequence_length: the token sequence length.
            max_norm: the maximum norm for back propagation.
            max_init_param: the maximum weight after initialization.
            device: the device on which the model will be. (either "cpu" or "gpu")
            sequence_step_size: the step size for batching (the smaller it is, the more overlap).
            lstm_configuration: the configuration of the lstm.
        """
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.number_of_layers = number_of_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_norm = max_norm
        self.max_init_param = max_init_param
        self.model_type = model_type

        if sequence_step_size is None:
            self.sequence_step_size = sequence_length
        else:
            self.sequence_step_size = sequence_step_size

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Set up the architecture.
        self.embedding = nn.Embedding(dictionary_size, embedding_size)

        if model_type == 'lstm':
            self.model_module = LSTM(self.embedding_size
                                     , number_of_layers
                                     , dropout_probability
                                     , lstm_configuration)

        elif model_type == 'ft':
            # TODO: FT class
            self.model_module = FT()

        else:
            self.model_module = None

        self.dropout = nn.Dropout(dropout_probability)

        # set initial weights randomly
        for param in self.parameters():
            nn.init.uniform_(param, -max_init_param, max_init_param)
        # set initial weights from pre-trained vectors
        if embedding_vectors is not None:
            self.embedding = prep_embedding_layer(vectors=embedding_vectors, trainable=embedding_trainable)

    def forward(self, X, states=None):
        X = self.embedding(X)
        # FT THING HERE
        # [1265, 26]


        if self.model_type == 'lstm':
            X = self.dropout(X)
            X, states = self.model_module(X, states)
            output = torch.tensordot(X, self.embedding.weight, dims=([2], [1]))
            return output, states

        elif self.model_type == 'ft':
            # TODO: FT class
            X = self.model_module(X)
            # some kind of linear thing here
            # some kind of ReLu activation stuff here
            # return some output and states here
