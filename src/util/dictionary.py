"""Dictionary related utils.
"""
from src.util import constants

from collections import Counter
from os.path import join

import json
import numpy as np

def dictionary_file_path(topic):
    """Returns a path to a dictionary file.

    Args:
        topic: the topic string.

    Returns:
        A file path.
    """
    return join(constants.DATA_PATH, "{}.dictionary.json".format(topic))

def get_dictionary_from_tokens(tokens, frequency_threshold):
    """Helper function that creates dictionary based on the tokens.

    Args:
        tokens: the tokens.
        frequency_threshold: minimum required frequency to be in the dictionary.

    Returns:
        A dictionary with (id, frequency) as values.
    """
    counter = Counter(tokens)
    dictionary = {}
    id = 0
    unk_count = 0
    for token in counter:
        if counter[token] >= frequency_threshold:
            dictionary[token] = (id, counter[token])
            id += 1
        else:
            unk_count += 1

    if "<unk>" in dictionary:
        id, count = dictionary["<unk>"]
        dictionary["<unk>"]= (id, count + unk_count)
    else:
        dictionary["<unk>"] = (id, unk_count)

    return dictionary

def change_to_unk(dictionary, tokens):
    """Changes tokens that are not in the dictionary to <unk>.

    Args:
        dictionary: the dictionary.
        tokens: the tokens.

    Returns:
        Number of tokens changed.
    """
    count = 0
    for i, token in enumerate(tokens):
        if not token in dictionary:
            tokens[i] = "<unk>"
            count += 1
    return count

def apply_dictionary_to_tokens(dictionary, tokens):
    """Applies the dictionary to the tokens.

    Args:
        dictionary: the dictionary.
        tokens: the tokens.

    Returns:
        Integer tokens.
    """
    return [dictionary[t][0] for t in tokens]

def get_glove_embeddings(dict_fn, dim):
    print("Getting Glove Embeddings")
    glove = {}
    with open('data/glove_6b/glove.6B.' + str(dim) + 'd.txt', 'r') as f:
        for l in f:
            line = l.split()
            glove[line[0]] = line[1:]

    with open(dict_fn) as jf:
        vocab = json.loads(jf.read())

    emb = np.zeros((len(vocab), dim))
    print(vocab)

    i = 0
    for v, [id, count] in vocab.items():
        v = v.lower()
        if v in glove:
            emb[id, :] = glove[v]
        else:
            emb[id, :] = np.random.rand(dim)
    return emb