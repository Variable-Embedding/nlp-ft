"""LSTM model set-up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import progressbar
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

def batch_data(tokens, model, batch_size=None, sequence_length=None, sequence_step_size=None):
    """Helper function to batch the data.

    Args:
        data: the data to batch.
        model: the model to batch for.
        batch_size: the batch size, if None will use model.batch_size.

    Returns:
        Iterator for batched data.
    """
    if batch_size is None:
        batch_size = model.batch_size
    if sequence_length is None:
        sequence_length = model.sequence_length
    if sequence_step_size is None:
        sequence_step_size = model.sequence_step_size

    data = torch.tensor(tokens, dtype=torch.int64).to(model.device)
    words_per_batch = data.size(0) // batch_size
    data = data[:words_per_batch * batch_size]
    data = data.view(batch_size, -1)

    for sequence_start in range(0, data.size(1) - sequence_length - 1, sequence_step_size):
        sequence_end = sequence_start + sequence_length
        prefix = data[:,sequence_start:sequence_end].transpose(1, 0)
        target = data[:,sequence_start + 1:sequence_end + 1].transpose(1, 0)
        yield prefix, target
    del data

def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)

def train_model(model, train_tokens, valid_tokens=None, number_of_epochs=1, logger=None):
    """Train the model in the train data.

    Args:
        model: the model to train.
        train_tokens: integer tokens for training.
        valid_tokens: integer tokens for validation.
        number_of_epochs: the number of epochs to run.
        logger: logger to use for printing output.

    Returns:
        A list of validation losses for each epoch (if validation tokens were provided).
    """
    model.to(model.device)
    num_iters = len(train_tokens) // model.batch_size // model.sequence_step_size
    training_losses = []
    validation_losses = []

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
            states = generate_initial_states(model)
            for prefix, target in batch_data(train_tokens, model):
                counter += 1
                model.zero_grad()
                states = detach_states(states)
                output, states = model(prefix, states)
                loss = loss_function(output, target)
                t_losses.append(loss.item() / model.batch_size)
                loss.backward()
                with torch.no_grad():
                    norm = nn.utils.clip_grad_norm_(model.parameters(), model.max_norm)
                    for param in model.parameters():
                        lr = model.learning_rate * (model.learning_rate_decay ** epoch)
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
    def __init__(self, embedding_size, hidden_size, number_of_layers, dropout_probability,
                 lstm_configuration):
        """Initializetion for LSTM module.

        Args:
            embedding_size: number of features in the embedding space.
            hidden_size: the size of hidden state.
            number_of_layers: number of LSTM layers (for stacked-LSTM).
            droupout_probability: the probability for dropping individual node in the network.
            lstm_configuration: the configuration of the lstm.
        """
        super().__init__()
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers=number_of_layers,
                            dropout=dropout_probability)

    def forward(self, X, states=None):
        X, states = self.lstm(X, states)
        return X, states


class Model(nn.Module):
    def __init__(self, dictionary_size, embedding_size=10, hidden_size=None, number_of_layers=1,
                 max_norm=0.0001, dropout_probability=0.1, batch_size=64, sequence_length=5,
                 learning_rate=0.0001, max_init_param=0.01, device="cpu", sequence_step_size=None,
                 learning_rate_decay=1, lstm_configuration="default"):
        """Initialization for the model.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
            hidden_size: the size of hidden state.
            number_of_layers: number of LSTM layers.
            max_norm: the maximum norm for the backward propagation.
            droupout_probability: the probability for dropping individual node in the network.
            batch_size: the batch size for the model.
            sequence_length: the token sequence length.
            learning_rate: the learning rate.
            max_init_param: the maximum weight after initialization.
            device: the device on which the model will be. (either "cpu" or "gpu")
            learning_rate_decay: learning rate decay
            sequence_step_size: the step size for batching (the smaller it is, the more overlap).
            lstm_configuration: the configuration of the lstm.
        """
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size if hidden_size is None else hidden_size
        self.number_of_layers = number_of_layers
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_init_param = max_init_param
        self.learning_rate_decay = learning_rate_decay

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
        self.lstm = LSTM(self.embedding_size, self.hidden_size, number_of_layers,
                         dropout_probability, lstm_configuration)
        self.fc = nn.Linear(self.hidden_size, dictionary_size)

        # Set initial weights.
        for param in self.parameters():
            nn.init.uniform_(param, -max_init_param, max_init_param)

    def forward(self, X, states=None):
        X = self.embedding(X)
        X, states = self.lstm(X, states)
        output = self.fc(X)
        return output, states
