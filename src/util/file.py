"""File related utils.
"""
from src.util import constants

from os.path import join

import json
import pickle

def get_tokens_from_file(file_path):
    """Helper function for getting tokens from file.

    Args:
        file_path: a path to the file.

    Returns:
        A list of tokens.
    """
    with open(file_path, "r") as file:
        text = file.read()
        tokens = text.split(" ")
    return tokens

def save_tokens_to_file(tokens, file_path):
    """Helper function for saving tokens to file.

    Args:
        tokens: a list of tokens.
        file_path: a path to the file.
    """
    with open(file_path, "w") as file:
        text = " ".join([str(t) for t in tokens])
        file.write(text)

def get_integer_tokens_from_file(file_path):
    """Helper function for getting integer tokens from file.

    Args:
        file_path: a path to the file.

    Returns:
        A list of tokens.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)

def save_integer_tokens_to_file(tokens, file_path):
    """Helper function for saving tokens to file.

    Args:
        tokens: a list of integer tokens.
        file_path: a path to the file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(tokens, file)

def get_dictionary_from_file(file_path):
    """Helper function for loading dictionary from file.

    Args:
        file_path: a path to file.
    """

def save_dictionary_to_file(dictionary, file_path):
    """Helper function for saving dictionary to file.

    Args:
        dictionary: the dictionary.
        file_path: a path to file.
    """


