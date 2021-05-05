"""LSTM model set-up.
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self
                 , embedding_size
                 , number_of_layers
                 , dropout_probability
                 , lstm_configuration
                 ):
        """Initialization for LSTM module.

        Args:
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers (for stacked-LSTM).
            dropout_probability: the probability for dropping individual node in the network.
            lstm_configuration: the configuration of the lstm. Possible configurations:
            Name            Description
            default         The regular stacked-lstm architecture
            var-emb         The input and states are passed through forward network before lstm
                            layer
            res-var-emb     The input and states are passed through forward and appended with input
        """
        super().__init__()
        configurations = {
            "default": 0,
            "att-emb": 1,
            "res-att-emb": 2,
            "ff-emb": 3,
            "res-ff-emb": 4,
        }
        self.configuration = configurations[lstm_configuration]

        if self.configuration != 0:
            self.ff = nn.Sequential(
                nn.Linear((1 + 2 * number_of_layers) * embedding_size, 3 * embedding_size),
                nn.ReLU(), nn.Dropout(dropout_probability),
                nn.Linear(3 * embedding_size, 3 * embedding_size),
                nn.ReLU(), nn.Dropout(dropout_probability),
                nn.Linear(3 * embedding_size, 3 * embedding_size),
                nn.ReLU(), nn.Dropout(dropout_probability),
                nn.Linear(3 * embedding_size, 2 * embedding_size),
                nn.ReLU(), nn.Dropout(dropout_probability),
                nn.Linear(2 * embedding_size, embedding_size)
            )

        if self.configuration == 2 or self.configuration == 4:
            self.lstm = nn.LSTM(2*embedding_size, embedding_size, num_layers=number_of_layers,
                                dropout=dropout_probability)
        else:
            self.lstm = nn.LSTM(embedding_size
                                , embedding_size
                                , num_layers=number_of_layers
                                , dropout=dropout_probability)

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, X, states=None):
        if self.configuration == 0:
            X = self.dropout(X)
            X, states = self.lstm(X, states)
        else:
            batch_size = X.shape[1]
            for i in range(X.shape[0]):
                H, C = states
                X_ = torch.cat((X[i].view(1, batch_size, -1), H.view(1, batch_size, -1),
                                      C.view(1, batch_size, -1)), 2)
                X_ = self.dropout(X_)
                X_ = self.ff(X_.view(1, batch_size, -1))

                # Attention-like mechanism
                if self.configuration == 2 or self.configuration == 1:
                    X_ = X_ + X[i].clone().view(1, batch_size, -1)

                # Residual-like mechanism
                if self.configuration == 2 or self.configuration == 4:
                    X_ = torch.cat((X[i].view(1, batch_size, -1), X_), 2)

                X_ = self.dropout(X_)
                X[i], states = self.lstm(X_, states)
        return X, states


