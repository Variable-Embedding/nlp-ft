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

    return [
        (torch.zeros(1, batch_size, layer.hidden_size, device=model.device),
         torch.zeros(1, batch_size, layer.hidden_size, device=model.device))
        for layer in model.rnns
    ]

def detach_states(states):
    """Helper function for detaching the states.

    Args:
        states: states to detach.

    Returns:
        List of detached states.
    """
    return [(h.detach(), c.detach()) for h, c in states]

def batch_data(tokens, model, batch_size=None, sequence_length=None):
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
    data = torch.tensor(tokens, dtype=torch.int64).to(model.device)
    num_batches = data.size(0) // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, -1)
    for sequence_start in range(0, data.size(1) - sequence_length, model.sequence_step_size):
        sequence_end = sequence_start + sequence_length
        prefix = data[:,sequence_start:sequence_end].transpose(1, 0)
        target = data[:,sequence_start + 1:sequence_end + 1].transpose(1, 0)
        yield prefix, target

def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    batch_size = target.size(1)
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * batch_size
    #probabilities = F.softmax(output, dim=2).reshape(-1, output.size(2))
    #target_probabilities = probabilities[range(target.numel()), target.reshape(-1)]
    #return torch.mean(-torch.log(target_probabilities) * batch_size)

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
            logger.info("Epoch #0, Validation preplexity: {:.1f}".format(validataion_loss))
        validation_losses.append(validataion_loss)

    counter = 0
    with progressbar.ProgressBar(max_value = number_of_epochs * num_iters) as progress_bar:
        progress_bar.update(0)
        for epoch in range(number_of_epochs):
            t_losses = []
            states = generate_initial_states(model)
            model.train()
            for prefix, target in batch_data(train_tokens, model):
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

                counter += 1
                progress_bar.update(counter)
                del prefix
                del target

            training_losses.append(np.exp(np.mean(t_losses)))
            if not valid_tokens is None:
                validataion_loss = test_model(model, valid_tokens)
                if not logger is None:
                    logger.info("Epoch #{}, Validation preplexity: {:.1f}".format(epoch + 1, validataion_loss))
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
    for prefix, target in batch_data(tokens, model):
        output, states = model(prefix, states)
        losses.append(loss_function(output, target).item() / model.batch_size)
        del prefix
        del target
    return np.exp(np.mean(losses))

def complete_sequence(model, prefix_tokens, sequence_end_token):
    """Using rnn language model, to complete the sequence.

    Args:
        model: the trained rnn language model.
        prefix_tokens: the start of the sequence.
        sequence_end_token: the token that specifies the end of the sequence.

    Returns:
        A list of tokens that go after the prefix tokens until sequence end token.
    """
    state = generate_initial_states(model, 1)
    prefix_tokens = torch.tensor(prefix_tokens)
    model.eval()
    # TODO(someone): implement this.
    return [sequence_end_token]


class Model(nn.Module):
    def __init__(self, dictionary_size, embedding_size=10, number_of_layers=1, max_norm=0.0001,
                 dropout_probability=0.1, batch_size=64, sequence_length=5, learning_rate=0.0001,
                 max_init_param=0.01, device="cpu", sequence_step_size=None, learning_rate_decay=1):
        """Initialization for the model.

        Args:
            dictionary_size: number of words in the dictionary.
            embedding_size: number of features in the embedding space.
            number_of_layers: number of LSTM layers.
            max_norm: the maximum norm for the backward propagation.
            droupout_probability: the probability for dropping individual node in the network.
            batch_size: the batch size for the model.
            sequence_length: the token sequence length.
            learning_rate: the learning rate.
            max_init_param: the maximum weight after initialization.
            device: the device on which the model will be. (either "cpu" or "gpu")
            learning_rate_decay: learning rate decay
        """
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
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
            self.device =  torch.device("cpu")

        # Set up the architecture.
        self.embedding = nn.Embedding(dictionary_size, embedding_size)
        rnns = [nn.LSTM(embedding_size, embedding_size) for _ in range(number_of_layers)]
        self.rnns = nn.ModuleList(rnns)
        self.fc = nn.Linear(embedding_size, dictionary_size)
        self.dropout = nn.Dropout(p=dropout_probability)

        # Set initial weights.
        for param in self.parameters():
            nn.init.uniform_(param, -max_init_param, max_init_param)

    def forward(self, X, states):
        X = self.embedding(X)
        X = self.dropout(X)
        for i, rnn in enumerate(self.rnns):
            X, states[i] = rnn(X, states[i])
            X = self.dropout(X)
        output = self.fc(X)
        return output, states
