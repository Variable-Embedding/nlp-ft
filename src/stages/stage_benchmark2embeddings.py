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

import logging
import numpy as np


# TODO: leverage torchtext API to combine benchmarks with embeddings


class Benchmark2Embeddings(BaseStage):
    """Stage for retrieving pre-trained embeddings with torchtext api.
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

        return True
