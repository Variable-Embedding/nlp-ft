from collections import Counter
import constants
from os.path import join
import json

def get_tokens_from_file(file_path):
    """Helper function for getting tokens from file.

    Args:
        file_path: a path to the file.

    Returns:
        A list of tokens.
    """
    with open(file_path) as file:
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

def get_dictionary_from_tokens(tokens, frequency_threshold):
    counter = Counter(tokens)
    dictionary = {}
    id = 0
    unk_count = 0
    for token in counter:
        if counter[token] > frequency_threshold:
            dictionary[token] = (id, counter[token])
            id += 1
        else:
            unk_count += 1
    dictionary["<<unk>>"] = [id, unk_count]
    return dictionary

def get_training_vocab_list(topic = 'countries'):
    return list(get_training_vocab_dict(topic).keys())

def get_training_vocab_dict(topic = 'countries'):
    dictionary_file_path = join(constants.DATA_PATH,"{}.dictionary.json".format(topic))
    with open(dictionary_file_path) as file:
        dictionary = json.loads(file.read())
    return dictionary

def get_corpus_integer_representation(topic = 'countries', set = 'train'):
    file_path = join(constants.DATA_PATH, "{}.{}.txt".format(topic, set))
    return get_tokens_from_file(file_path)

# vocab_dict = get_training_vocab_dict()
# vocab_list = get_training_vocab_list()
# training_int_rep = get_corpus_integer_representation(set='train')
# validation_int_rep = get_corpus_integer_representation(set='valid')
# testing_int_rep = get_corpus_integer_representation(set='test')