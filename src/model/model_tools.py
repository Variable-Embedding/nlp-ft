from torch.nn import Embedding, utils

from torch import zeros, tensor, argmax, no_grad
import torch.nn.functional as F
import numpy as np
import progressbar


def train_model(model
                , train_tokens
                , valid_tokens=None
                , number_of_epochs=1
                , learning_rate=1
                , learning_rate_decay=1
                , logger=None
                ):
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
    loss_function = cross_entropy_loss

    if not logger is None:
        num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("Number of model parameters: {}".format(num_parameters))
        logger.info(f"Training with hardware: {model.device}")

    if not valid_tokens is None:
        validataion_loss = test_model(model, valid_tokens)
        if not logger is None:
            logger.info("Epoch #0, Validation perplexity: {:.1f}".format(validataion_loss))
        validation_losses.append(validataion_loss)

    counter = 0
    with progressbar.ProgressBar(max_value=number_of_epochs * num_iters) as progress_bar:
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
                    # FIXME: If trainable embedding is false, skip it here
                    with no_grad():
                        norm = utils.clip_grad_norm_(model.parameters(), model.max_norm)
                        for param in model.parameters():
                            if param.grad != None:
                                lr = learning_rate * (learning_rate_decay ** epoch)
                                param -= lr * param.grad

            training_losses.append(np.exp(np.mean(t_losses)))
            if not valid_tokens is None:
                validataion_loss = test_model(model, valid_tokens)
                if not logger is None:
                    logger.info("Epoch #{}, Validation perplexity: {:.1f}".format(epoch + 1,
                                                                                  validataion_loss))
                validation_losses.append(validataion_loss)
    return training_losses, validation_losses


def test_model(model, tokens, loss_function=None):
    """Test the model on the tokens.

    Args:
        model: model to test.
        tokens: the tokens to test the model on.

    Returns:
        Preplexity score of the model
    """
    if loss_function is None:
        loss_function = cross_entropy_loss

    model.to(model.device)
    losses = []
    states = generate_initial_states(model)

    model.eval()
    for prefix, target in batch_data(tokens, model, sequence_step_size=model.sequence_length):
        output, states = model(prefix, states)
        losses.append(loss_function(output, target).item() / model.batch_size)
    return model_perplexity(losses)


def model_perplexity(losses):
    """Return model perplexity of the mean of the losses.

    Args:
        losses: a list of floats.

    Returns: a float.

    """
    return np.exp(np.mean(losses))


def prep_embedding_layer(vectors, trainable=False):
    """A helper function to return pytorch nn embedding layer.

    Args:
        vectors: weight matrix of pre-trained or randomized vectors
        trainable: bool, default to False. If False, keep static.
    Returns:

        embedding_layer: torch sparse embedding layer, number of embeddings, and number of embedding dims

    source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """

    num_embeddings, embedding_dim = vectors.size()

    embedding_layer = Embedding(num_embeddings, embedding_dim)
    embedding_layer.load_state_dict({'weight': vectors})

    if trainable:
        embedding_layer.weight.requires_grad = True
    else:
        embedding_layer.weight.requires_grad = False

    return embedding_layer


def cross_entropy_loss(output, target):
    """A cross entropy function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        loss: cross entropy loss
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)


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

    return (zeros(model.number_of_layers, batch_size, model.embedding_size, device=model.device),
            zeros(model.number_of_layers, batch_size, model.embedding_size, device=model.device))


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

    # data = torch.tensor(tokens, dtype=torch.int64)
    data = tokens
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
    prefix_tokens = tensor(prefix_tokens).to(model.device)
    state = generate_initial_states(model, 1)
    result = []

    # Setting up hidden state based on the prefix tokens.
    output, state = model(prefix_tokens.reshape(-1, 1, 1), state)

    # TODO: add beam-search option here?
    # Generating the continuation of the sequence.
    curr_token = prefix_tokens[-1]
    num_tokens_left = max_sequence_length
    while curr_token != sequence_end_token and num_tokens_left > 0:
        curr_token = argmax(output, dim=2)
        result.append(curr_token.item())
        output, state = model(curr_token.reshape(1, 1, 1), state)
        num_tokens_left -= 1

    return result
