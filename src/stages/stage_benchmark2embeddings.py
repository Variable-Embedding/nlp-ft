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


def make_corpra_vocab(corpra_cache, logger, vectors_cache=None, min_freq=None, torch_tokenizer='basic_english'):
    counter = Counter()
    tokenizer = get_tokenizer(torch_tokenizer)
    min_freq = 1 if min_freq is None else min_freq

    vectors = Vectors(vectors_cache)

    for corpus_cache in corpra_cache:
        logger.info(f'Reading corpus cache from {corpus_cache}')
        f = open(corpus_cache, 'r')

        for line in f:
            counter.update(tokenizer(line))

    v = Vocab(counter, min_freq=min_freq, vectors=vectors, vectors_cache=vectors_cache)

    logger.info(f'Generated torch Vocab object with dictionary size of {len(v.stoi)}.')

    random_word = random.choice(v.itos)
    random_word_index = v.stoi[random_word]
    random_word_curr_vector = v.vectors[random_word_index]
    random_word_orig_vector = vectors[random_word]
    # the torch vocab object has mapped the vocab index to the embedding layer
    assert random_word_curr_vector.all() == random_word_orig_vector.all()

    return v


def corpra_caches(corpus_type, logger):
    corpus_type = 'wikitext-2' if "wikitext2" == corpus_type else "wikitext-103" if "wikitext103" == corpus_type else corpus_type
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
    cache_path = os.sep.join([constants.EMBEDDINGS_PATH, f'{embedding_type}.txt'])
    if os.path.exists(cache_path):
        logger.info(f"Found embedding cache at {cache_path}.")
        return cache_path


def read_pickle(cache_path, logger):
    logger.info(f'Opening file at {cache_path}. This may take a while')

    with open(cache_path, "rb") as handle:
        pickle_file = pickle.load(handle)

    return pickle_file


class Benchmark2Embeddings(BaseStage):
    """Stage for mapping benchmark corpra to pre-trained embeddings.
    """
    name = "benchmark2embeddings"
    logger = logging.getLogger("pipeline").getChild("benchmark2embeddings_stage")

    def __init__(self, parent=None, embedding_type=None, corpus_type=None):
        """Initialization for Benchmark 2 Embeddings Stage.
        """
        super().__init__(parent)
        self.embedding_type = 'glove.6B.100d' if embedding_type is None else embedding_type
        self.corpus_type = 'wikitext2' if corpus_type is None else corpus_type

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
        corpra_cache = corpra_caches(self.corpus_type, self.logger)

        vocab = make_corpra_vocab(corpra_cache=corpra_cache, vectors_cache=vectors_cache, logger=self.logger)

        return True
