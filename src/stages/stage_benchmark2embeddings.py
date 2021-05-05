"""Stage for combining pre-trained embeddings with benchmark corpra
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
from torchtext.vocab import Vocab, Vectors
from collections import Counter
import logging
import numpy as np
from torchtext.data.utils import get_tokenizer
import random
import torch


def make_corpra_vocab(logger, tokenizer, vectors_cache=None, min_freq=None, corpra_cache=None, corpra_object=None,
                      corpus_type=None):
    """A helper function to create torchtext vocab objects from benchmark texts.
    Combines pre-trained embedding vectors with torch objects.

    Args:
        corpra_cache: a list of os paths to corpra, optional. If not provided, a torch vocab (corpra_object) is required.
        logger: a logging object
        tokenizer: a torchtext tokenizer object
        vectors_cache: an os path to the pre-trained embedding
        min_freq: an integer such as 1 or 5, If none, value is 1.
        corpra_type: a string, name of the corpus


    Returns: v, a torchtext objecting having global vocabulary, lookup tables, and embedding layers

    """
    logger.info('Starting to parse corpra into vocab and iterable objects. This may take a while.')
    corpra = {}
    counter = Counter()
    min_freq = 1 if min_freq is None else min_freq
    logger.info(f'Loading vectors from {vectors_cache}.')
    vectors = Vectors(vectors_cache)

    # forcing imdb to run from corpus object
    if corpra_cache is not None and 'imdb' not in corpus_type:

        for corpus_cache in corpra_cache:
            logger.info(f'Reading corpus cache from {corpus_cache}')
            key = 'train' if '.train.' in corpus_cache else 'test' if '.test.' in corpus_cache else 'valid'
            corpus = []
            f = open(corpus_cache, 'r')
            logger.info(f'Tokenizing and making vocabulary for {key} set.')
            for line in f:
                counter.update(tokenizer(line))
                corpus.extend(tokenizer(line))
            corpra.update({key: corpus})

    elif corpra_object is not None:

        def corpra_key(x, o):
            if len(o) == 2:
                return 'train' if x == 0 else 'test'
            else:
                return 'train' if x == 0 else 'valid' if x == 1 else 'test'

        for idx, corpus_object in enumerate(corpra_object):
            key = corpra_key(idx, corpra_object)
            corpus = []
            logger.info(f'Tokenizing and making vocabulary for {key} set.')

            if corpus_type == 'imdb':

                for line in corpus_object:
                    tokens = tokenizer(line[1])
                    counter.update(tokens)
                    labels_tokens = tuple((line[0], tokens))
                    corpus.append(labels_tokens)
                corpra.update({key: corpus})
            else:

                for line in corpus_object:
                    counter.update(tokenizer(line))
                    corpus.extend(tokenizer(line))
                corpra.update({key: corpus})

    v = Vocab(counter, min_freq=min_freq, vectors=vectors, vectors_cache=vectors_cache)

    text_pipeline = lambda x: [v[token] for token in tokenizer(x)]
    label_code = lambda x: 0 if x == 'neg' else 1 if x == 'pos' else 2
    corpra_numeric = {}
    corpra_labels = {}

    for data_set, corpus in corpra.items():
        logger.info(f'Converting string tokens to numeric tokens for {data_set}.')
        corpus_numeric = []
        corpus_labels = {}
        if corpus_type == "imdb":
            for idx, line in enumerate(corpus):
                tokens = str(line[1])
                label = torch.tensor(label_code(str(line[0])), dtype=torch.long)
                numeric_tokens = torch.tensor(text_pipeline(tokens), dtype=torch.long)
                # labels_tokens = tuple((label, numeric_tokens))
                corpus_numeric.append(numeric_tokens)
                # idx_labels = tuple((idx, label))
                corpus_labels.update({idx: label})

            corpra_numeric.update({data_set: corpus_numeric})
            corpra_labels.update({data_set: corpus_labels})

        else:

            for line in corpus:
                numeric_tokens = text_pipeline(line)
                corpus_numeric.extend(numeric_tokens)

            corpus_numeric = torch.tensor(corpus_numeric, dtype=torch.long)
            corpra_numeric.update({data_set: corpus_numeric})

    logger.info(f'Generated torch Vocab object with dictionary size of {len(v.stoi)}.')

    random_word = random.choice(v.itos)
    random_word_index = v.stoi[random_word]
    random_word_curr_vector = v.vectors[random_word_index]
    random_word_orig_vector = vectors[random_word]
    # the torch vocab object has mapped the vocab index to the embedding layer
    assert random_word_curr_vector.all() == random_word_orig_vector.all()

    if corpus_type == 'imdb':
        return v, corpra_numeric, corpra_labels
    else:
        return v, corpra_numeric


def make_benchmark_corpra(vocab, tokenizer, cache_paths=None, corpra_object=None):
    """Leveraging torchtext functions.
    Args:
        corpra_object: optional but strongly recommended, if passed, provide a torchtext object of vocab
        cache_paths: optional, if passed, provide a list of os paths to locations of text
        vocab: a torchtext vocab object
        tokenizer: a torchtext tokenizer object
    Returns:
        corpus: a dictionary of corpra keyed by 'train', 'valid', and 'test' stages
    """

    corpra = {}
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    if cache_paths is not None:
        for cache_path in cache_paths:
            key = 'train' if '.train.' in cache_path else 'test' if '.test.' in cache_path else 'valid'
            f = open(cache_path, 'r')
            corpus = []

            for line in f:
                c = text_pipeline(line)
                corpus.extend(c)

            corpus = torch.tensor(corpus, dtype=torch.long)
            corpra.update({key: corpus})

            f.close()

    elif corpra_object is not None:
        # FIXME: WTF this?
        for idx, corpus_object in enumerate(corpra_object):
            key = 'train' if idx == 0 else 'valid' if idx == 1 else 'test'
            for i in corpus_object:
                print(i)

    return corpra


def corpra_caches(logger, corpus_type=None):
    """A helper function to return a series of os paths to text caches

    Args:
        corpus_type: the name of a benchmark corpus -> like "wikitext2" or "imdb"
        logger: a logging object

    Returns: cache_paths, a list of os paths to benchmark text corpra on disk.

    """

    folder_names = {
        'wikitext2': 'wikitext-2'
        , 'wikitext103': 'wikitext-103'
        , 'imdb': 'aclimdb'}

    if any(corpus_type in i for i in folder_names.keys()):
        corpus_folder_name = folder_names[corpus_type]
        corpus_type = corpus_type.replace(corpus_type, corpus_folder_name)

    cache_path = os.sep.join([constants.DATA_PATH, corpus_type])

    cache_paths = []

    if os.path.exists(cache_path):
        logger.info(f"Found corpra cache at {cache_path}.")
        files = os.listdir(cache_path)

        for file in files:
            if file.endswith(".tokens") or file.endswith(".txt"):
                corpus_path = os.sep.join([cache_path, file])
                cache_paths.append(corpus_path)

        return cache_paths


def embedding_cache(embedding_type, logger):
    """A helper function to return an os path to a specified cache location.

    Args:
        embedding_type: a string, the name of the target embedding type -> like "glove.6B"
        logger: a logging object

    Returns: an os path string

    """
    cache_path = os.sep.join([constants.EMBEDDINGS_PATH, f'{embedding_type}.txt'])
    if os.path.exists(cache_path):
        logger.info(f"Found embedding cache at {cache_path}.")
        return cache_path


def read_pickle(cache_path, logger):
    """A helper function to read pickled data from disk.

    Args:
        cache_path: an os path to a pickle file to be opened.
        logger: a logging object

    Returns: a pickled file object

    """

    logger.info(f'Opening file at {cache_path}. This may take a while')

    with open(cache_path, "rb") as handle:
        pickle_file = pickle.load(handle)

    return pickle_file


class Benchmark2Embeddings(BaseStage):
    """Stage for mapping benchmark corpra to pre-trained embeddings.
    """
    name = "benchmark2embeddings"
    logger = logging.getLogger("pipeline").getChild("benchmark2embeddings_stage")

    def __init__(self
                 , parent=None
                 , embedding_type=None
                 , corpus_type=None
                 , corpra_object=None
                 , tokenizer=None
                 , min_freq=1
                 ):
        """Initialization for Benchmark 2 Embeddings Stage.
        """
        super().__init__(parent)
        self.embedding_type = 'glove.6B.100d' if embedding_type is None else embedding_type
        self.corpus_type = corpus_type
        self.vocab = None
        self.corpra_cache = None
        self.tokenizer = get_tokenizer('basic_english') if tokenizer is None else tokenizer
        self.corpra_object = corpra_object
        self.corpra_numeric = None
        self.min_freq = min_freq
        self.corpra_labels = None

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing get pretrained embedding stage")
        self.logger.info("-" * 40)

    def run(self):
        """Combines text benchmarks with pre-trained embedding spaces.

        :return: True if the stage execution succeeded, False otherwise.
        """

        vectors_cache = embedding_cache(self.embedding_type, self.logger)
        if self.corpus_type is not None:
            self.corpra_cache = corpra_caches(corpus_type=self.corpus_type, logger=self.logger)

        res = make_corpra_vocab(corpra_cache=self.corpra_cache
                                , vectors_cache=vectors_cache
                                , logger=self.logger
                                , tokenizer=self.tokenizer
                                , corpra_object=self.corpra_object
                                , min_freq=self.min_freq
                                , corpus_type=self.corpus_type)

        if len(res) < 3:
            self.vocab = res[0]
            self.corpra_numeric = res[1]
        else:
            self.vocab = res[0]
            self.corpra_numeric = res[1]
            self.corpra_labels = res[2]

        return True
