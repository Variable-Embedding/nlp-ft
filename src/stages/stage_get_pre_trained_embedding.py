"""Stage for retrieving and processing pre-trained embeddings.
"""
from src.stages.base_stage import BaseStage
from src.util import constants

import os
from tqdm import tqdm
import zipfile
import pickle
import time
import mmap
from pathlib import Path
from torchtext.vocab import pretrained_aliases

import logging
import numpy as np
import requests


def download_embeddings(embedding_alias, embedding_type, logger):
    """Get and process GloVe embeddings.
    """
    if "." in embedding_type:
        embedding_type = embedding_type.replace(".", '_')

    embedding_cache = os.sep.join([constants.EMBEDDINGS_PATH, embedding_type])

    if not os.path.exists(embedding_cache):
        os.makedirs(embedding_cache)

    directory = os.listdir(embedding_cache)

    if len(directory) > 0 and any(".txt" in i for i in directory):
        logger.info(f'Directory Not Empty {embedding_cache}.')
        logger.info(f'Current files include: {directory}')
        logger.info(f'Skipping download. To force download, delete or rename files in this directory.')

        write_pickle(directory, embedding_cache)

    else:
        # source: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        if not any(".zip" in i for i in directory):
            logger.info(f'Downloading embedding data for {embedding_type} to {embedding_cache}')
            embedding_alias(cache=embedding_cache)

            directory = os.listdir(embedding_cache)

            write_pickle(directory, embedding_cache, logger)

    return True


def write_pickle(directory, unzip_path, logger):
    """Write dictionaries to pickled files if the provided directory does not already have pickle files.

    :param directory: A dictionary of word embeddings.
    :param unzip_path: Full path to file locations.
    :param logger: the logging object for this process.
    :return: None, writes data to disk.
    """

    if any([".pickle" in i for i in directory]):
        logger.info(f'Pickle files detected, skipping conversion from .txt to .pickle format.')

    else:
        logger.info(f'Converting .txt embedding files to .pickle objects for faster IO.')

        for text_file in directory:
            text_file_path = os.sep.join([unzip_path, text_file])

            if text_file_path.endswith(".txt"):

                embedding_dict = parse_embedding_txt(text_file_path, logger)

                logger.info(f'Parsing embedding files as pickles. This may take a while.')

                pickle_file = f'{Path(text_file).stem}.pickle'
                pickle_file_path = os.sep.join([unzip_path, pickle_file])

                # TODO: Add a spinning pbar or something here
                f = open(pickle_file_path, "wb")
                pickle.dump(embedding_dict, f, protocol=2)
                f.close()

                # test read pickle
                parse_embedding_pickle(pickle_file_path, logger)

                logger.info(f'Wrote embedding pickle to {pickle_file_path}.')


def parse_embedding_pickle(embedding_file_path, logger):
    """Read pickled embedding files from disk.

    :param embedding_file_path: string, full path to the embedding txt file
    :param logger: the logging object for this process.
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logger.info(f'Reading embedding file from {embedding_file_path}.')

    if ".pickle" in embedding_file_path:
        start_time = time.time()

        with open(embedding_file_path, "rb") as handle:
            embedding_dict = pickle.load(handle)

        end_time = time.time()
        logger.info(f'PICKLE Read Time is {end_time - start_time}')
    else:
        logger.info(f'Did not read file, please indicate a file ending in .pickle.')

    return embedding_dict


def parse_embedding_txt(embedding_file_path, logger):
    """Read text embedding files from disk.

    :param embedding_file_path: string, full path to the embedding txt file
    :param logger: the logging object for this process.
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logger.info(f'Parsing word embeddings from {embedding_file_path}')
    embedding_dict = {}
    embedding_dim = 300 if "300d" in embedding_file_path else 200 if "200d" in embedding_file_path else 100 if "100d" in embedding_file_path else 50 if "50d" in embedding_file_path else 25

    start_time = time.time()
    with open(embedding_file_path, 'r', encoding="utf-8") as f:
        # for line in f:
        for line in tqdm(f, total=get_num_lines(embedding_file_path), desc='Opening Embedding File'):
            word, vector = read_line(line, embedding_dim=embedding_dim)
            embedding_dict.update({word: vector})
    end_time = time.time()
    logger.info(f'TEXT Read Time is {end_time-start_time}')

    return embedding_dict


def get_num_lines(file_path):
    """A helper function to count number of lines in a given text file.

    :param file_path: full path to some .txt file.
    :return: integer, count of lines in a .txt file.

    reference: https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
    """

    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def read_line(line, embedding_dim):
    """Read lines in a text file from embeddings.

    :param line: Each line of a text open object.
    :param embedding_dim: the expected vector dimension
    :return: 2-tuple of word (string) and numpy array (vector).
    """
    first = 0
    rest = 1
    values = line.split()
    # the first element is assumed to be the word
    word = values[first]

    # catch cases where first n strings are repeating as the word
    try:
        # the rest of list is usually the vector but sometimes it is not
        vector = np.asarray(values[rest:], "float32")

    except ValueError:
        #TODO: words such as ... or -0.0033421 are truncating the vector space from 300 to less than 300,
        # impacting about a dozen entries, provide random vector for now, fix later
        word, rest = return_repeating_word(values)
        vector = random_embedding_vector(embedding_dim=embedding_dim)

    if len(vector) != embedding_dim:
        vector = random_embedding_vector(embedding_dim=embedding_dim)

    return tuple((word, vector))


def return_repeating_word(values):
    """A helper function for read_line().

    Address issues where word has repeating chargers, return them as a single word.

    :param values: values, a line of embedding text data
    :return: A string of repeating characters.
    """
    word = []
    first_char = values[0]
    counter = 1
    word.append(first_char)

    # while values:
    for idx, char in enumerate(values[1:]):
        counter += 1
        curr_char = char
        if curr_char == first_char:
            word.append(curr_char)
        else:
            break

    word = ''.join(map(str, word))

    return word, counter


def random_embedding_vector(embedding_dim, scale=0.6):
    """A helper function to return a randomized embedding space of dimension embedding_dim

    :param embedding_dim: integer, usually 300 or one of 200, 100, 50, 25, depending on embedding space.
    :param scale: stdev of distribution for np.random.normal function
    :return: a randumized numpy array to fill an embedding vector
    """
    return np.random.normal(scale=scale, size=(embedding_dim,))


def get_embedding_alias(embedding_type):
    embedding_aliases = pretrained_aliases.keys()
    if embedding_type in embedding_aliases:
        return pretrained_aliases[embedding_type]
    else:
        return False


class GetPreTrainedEmbeddingsStage(BaseStage):
    """Stage for scraping the data from the wikipedia.
    """
    name = "get_pre_trained_embeddings"
    logger = logging.getLogger("pipeline").getChild("get_pre_trained_embeddings_stage")

    def __init__(self, parent=None, embedding_type=None):
        """Initialization for Get Pre-Trained Embedding Stage.
        """
        super().__init__(parent)
        self.embedding_type = 'glove.6B.100d' if embedding_type is None else embedding_type

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing get pretrained embedding stage")
        self.logger.info("-" * 40)

    def run(self):
        """Retrieves pre-trained embeddings from various sources.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        embedding_alias = get_embedding_alias(self.embedding_type)

        if embedding_alias is False:
            self.logger.info(f'Provided embedding type of '
                             f'"{self.embedding_type}" not found,'
                             f' pick from any of the following: {list(pretrained_aliases.keys())}')
        else:
            download_embeddings(embedding_alias, self.embedding_type, self.logger)

        return True