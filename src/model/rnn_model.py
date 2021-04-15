"""LSTM model set-up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import progressbar
from src.util.dictionary import get_glove_embeddings
import time


def generate_initial_states(model, batch_size=None):
    """Helper function to generate initial state needed for the model.forward function

    Args:
        model: model for which the states are initialized.
        batch_size: the batch size for states. If None will use model.batch_size.

    Returns:
        A list of tuples of torch.array.
    """
    if batch_size is None:
        batch_size = model.batch_size

    return (torch.zeros(model.number_of_layers, batch_size, model.embedding_size,
                        device=model.device),
            torch.zeros(model.number_of_layers, batch_size, model.embedding_size,
                        device=model.device))

def detach_states(states):
    """Helper function for detaching the states.

    Args:
        states: states to detach.

    Returns:
        List of detached states.
    """
    h, c = states
    return (h.detach(), c.detach())

def batch_data(tokens, model, batch_size=None, sequence_length=None, sequence_step_size=None,
               shuffle=False):
    """Helper function to batch the data.

    Args:
        data: the data to batch.
        model: the model to batch for.
        batch_size: the batch size, if None will use model.batch_size.
        sequence_step_size: the sequence step size.
        shuffle: Whether to shuffle the order of sequences.

    Returns:
        Iterator for batched data.
    """
    if batch_size is None:
        batch_size = model.batch_size
    if sequence_length is None:
        sequence_length = model.sequence_length
    if sequence_step_size is None:
        sequence_step_size = model.sequence_step_size

    data = torch.tensor(tokens, dtype=torch.int64)
    words_per_batch = data.size(0) // batch_size
    data = data[:words_per_batch * batch_size]
    data = data.view(batch_size, -1)

    sequence_start_list = list(range(0, data.size(1) - sequence_length - 1, sequence_step_size))
    if shuffle:
        np.random.shuffle(sequence_start_list)

    for sequence_start in sequence_start_list:
        sequence_end = sequence_start + sequence_length
        prefix = data[:,sequence_start:sequence_end].transpose(1, 0).to(model.device)
        target = data[:,sequence_start + 1:sequence_end + 1].transpose(1, 0).to(model.device)
        yield prefix, target
        del prefix
        del target

def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)

def train_model(model, train_tokens, valid_tokens=None, number_of_epochs=1,
                learning_rate=1, learning_rate_decay=1, logger=None):
    """Train the model in the train data.

    Args:
        model: the model to train.
        train_tokens: integer tokens for training.
        valid_tokens: integer tokens for validation.
        number_of_epochs: the number of epochs to run.
        learning_rate: the learning rate.
        learning_rate_decay: learning rate decay
        logger: logger to use for printing output.

    Returns:
        A list of validation losses for each epoch (if validation tokens were provided).
    """
    model.to(model.device)
    num_iters = len(train_tokens) // model.batch_size // model.sequence_step_size
    num_iters += len(train_tokens) // model.batch_size // model.sequence_length
    training_losses = []
    validation_losses = []

    if not logger is None:
        num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("Number of model parameters: {}".format(num_parameters))

    if not valid_tokens is None:
        validataion_loss = test_model(model, valid_tokens)
        if not logger is None:
            logger.info("Epoch #0, Validation perplexity: {:.1f}".format(validataion_loss))
        validation_losses.append(validataion_loss)

    counter = 0
    with progressbar.ProgressBar(max_value = number_of_epochs * num_iters) as progress_bar:
        for epoch in range(number_of_epochs):
            progress_bar.update(counter)
            t_losses = []
            model.train()
            for shuffle in [True, False]:
                states = generate_initial_states(model)
                step_size = model.sequence_step_size if shuffle else model.sequence_length
                for prefix, target in batch_data(train_tokens, model, sequence_step_size=step_size,
                                                 shuffle=shuffle):
                    progress_bar.update(counter)
                    counter += 1
                    model.zero_grad()
                    states = generate_initial_states(model) if shuffle else detach_states(states)
                    output, states = model(prefix, states)
                    loss = loss_function(output, target)
                    t_losses.append(loss.item() / model.batch_size)
                    loss.backward()
                    with torch.no_grad():
                        norm = nn.utils.clip_grad_norm_(model.parameters(), model.max_norm)
                        for param in model.parameters():
                            lr = learning_rate * (learning_rate_decay ** epoch)
                            # if param.grad:
                            print(type(param), param.size())
                            param -= lr * param.grad

            training_losses.append(np.exp(np.mean(t_losses)))
            if not valid_tokens is None:
                validataion_loss = test_model(model, valid_tokens)
                if not logger is None:
                    logger.info("Epoch #{}, Validation perplexity: {:.1f}".format(epoch + 1,
                                                                                  validataion_loss))
                validation_losses.append(validataion_loss)
    return training_losses, validation_losses

def test_model(model, tokens):
    """Test the model on the tokens.

    Args:
        model: model to test.
        tokens: the tokens to test the model on.

    Returns:
        Preplexity score of the model
    """
    model.to(model.device)
    losses = []
    states = generate_initial_states(model)
    model.eval()
    for prefix, target in batch_data(tokens, model, sequence_step_size=model.sequence_length):
        output, states = model(prefix, states)
        losses.append(loss_function(output, target).item() / model.batch_size)
    return np.exp(np.mean(losses))

def complete_sequence(model, prefix_tokens, sequence_end_token, max_sequence_length=1000):
    """Using lstm language model, to complete the sequence.

    Args:
        model: the trained rnn language model.
        prefix_tokens: the start of the sequence.
        sequence_end_token: the token that specifies the end of the sequence.
        max_sequence_length: the maximum length of the output sequence.

    Returns:
        A list of tokens that go after the prefix tokens until sequence end token.
    """
    model.to(model.device)
    model.eval()
    prefix_tokens = torch.tensor(prefix_tokens).to(model.device)
    state = generate_initial_states(model, 1)
    result = []

    # Setting up hidden state based on the prefix tokens.
    output, state = model(prefix_tokens.reshape(-1, 1, 1), state)

    # TODO: add beam-search option here?
    # Generating the continuation of the sequence.
    curr_token = prefix_tokens[-1]
    num_tokens_left = max_sequence_length
    while curr_token != sequence_end_token and num_tokens_left > 0:
        curr_token = torch.argmax(output, dim=2)
        result.append(curr_token.item())
        output, state = model(curr_token.reshape(1, 1, 1), state)
        num_tokens_left -= 1

    return result


class LSTM(nn.Module):
    def __init__(self, embedding_size, number_of_layers, dropout_probability,
                 lstm_configuration):
        """Initializetion for LSTM module.

        Args:
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers (for stacked-LSTM).
            droupout_probability: the probability for dropping individual node in the network.
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
            "ff-emb-plstm": 5,
        }
        self.configuration = configurations[lstm_configuration]

        if self.configuration != 0:
            self.ff = nn.Sequential(
                nn.Linear((1 + 2 * number_of_layers) * embedding_size, 3 * embedding_size),
                nn.Tanh(), nn.Dropout(dropout_probability),
                # nn.Linear(3 * embedding_size, 3 * embedding_size),
                # nn.Tanh(), nn.Dropout(dropout_probability),
                # nn.Linear(3 * embedding_size, 3 * embedding_size),
                # nn.Tanh(), nn.Dropout(dropout_probability),
                nn.Linear(3 * embedding_size, 2 * embedding_size),
                nn.Tanh(), nn.Dropout(dropout_probability),
                nn.Linear(2 * embedding_size, embedding_size),
                nn.Tanh()
            )

        if self.configuration == 2 or self.configuration == 4:
            self.lstm = nn.LSTM(2*embedding_size, embedding_size, num_layers=number_of_layers,
                                dropout=dropout_probability)
        if self.configuration == 5:
            self.plstm = nn.LSTM(embedding_size, embedding_size, num_layers=number_of_layers,
                                dropout=dropout_probability)
        else:
            self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers=number_of_layers,
                                dropout=dropout_probability)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, X, states=None):
        if self.configuration == 0:
            X = self.dropout(X)
            X, states = self.lstm(X, states)
        # elif self.configuration == 1:
        #     batch_size = X.shape[1]
        #     for i in range(X.shape[0]):
        #         H, C = states
        #         X_ = X_ * X[i].clone().view(1, batch_size, -1)
        #         X_ = self.dropout(X_)
        #         X[i], states = self.lstm(X_, states)
        else:
            batch_size = X.shape[1]
            for i in range(X.shape[0]):
                H, C = states
                # H, C = detach_states(states)
                X_ = torch.cat((X[i].view(1, batch_size, -1), H.view(1, batch_size, -1),
                                      C.view(1, batch_size, -1)), 2)
                X_ = self.dropout(X_)
                X_ = self.ff(X_.view(1, batch_size, -1))

                # Attention-like mechanism
                if self.configuration == 2 or self.configuration == 1:
                    X_ = X_ * X[i].clone().view(1, batch_size, -1)

                # Residual-like mechanism
                if self.configuration == 2 or self.configuration == 4:
                    X_ = torch.cat((X[i].view(1, batch_size, -1), X_), 2)

                X_ = self.dropout(X_)
                X[i], states = self.lstm(X_, states)
        return X, states


class Model(nn.Module):
    def __init__(self, dictionary_size, embedding_size=100, number_of_layers=1,
                 dropout_probability=0.3, batch_size=64, sequence_length=30, max_norm=2,
                 max_init_param=0.01, device="cpu", sequence_step_size=None,
                 lstm_configuration="default", embedding_config='default', freeze_embeddings = False):
        """Initialization for the model.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers.
            droupout_probability: the probability for dropping individual node in the network.
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
        self.lstm = LSTM(self.embedding_size, number_of_layers,
                         dropout_probability, lstm_configuration)
        self.dropout = nn.Dropout(dropout_probability)
        # self.pre_output = nn.Linear(self.embedding_size, self.embedding_size)

        # Set initial weights.
        for param in self.parameters():
            nn.init.uniform_(param, -max_init_param, max_init_param)

        if embedding_config == 'glove':
            emb = get_glove_embeddings("data/wiki.dictionary.json", embedding_size)
            self.embedding.weight.data.copy_(torch.from_numpy(emb))
            if freeze_embeddings: self.embedding.weight.requires_grad = False

        # self.fc = nn.Linear(embedding_size, dictionary_size)

    def forward(self, X, states=None):
        X = self.embedding(X)
        X = self.dropout(X)
        X, states = self.lstm(X, states)
        X = self.dropout(X)
        # X = self.pre_output(X)
        # print("x shape",X.shape)
        # print("embedding weight shape",self.embedding.weight.shape)
        output = torch.tensordot(X, self.embedding.weight, dims=([2], [1]))
        # print("output shape",self.embedding.weight.shape)
        # output = self.fc(output)
        return output, states
