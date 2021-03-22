"""Stage for pre-processing text.
"""
from src.model.rnn_model import train_model, test_model, Model, complete_sequence
from src.stages.base_stage import BaseStage
from src.util import constants
from src.util.dictionary import dictionary_file_path
from src.util.file import get_integer_tokens_from_file

from os.path import join

import json
import logging
import re
import termplotlib as tpl
import matplotlib.pyplot as plt
import torch
import yaml


class GenerateTextRnnModelStage(BaseStage):
    """Stage for using rnn model to generate text.
    """
    name = "generate_sequence_rnn_model"
    logger = logging.getLogger("pipeline").getChild("train_rnn_model")

    def __init__(self, parent=None, input_file=None, output_file=None):
        """Initialization for text generation stage.

        Args:
            parent: the parent stage.
            input_file: the file with prefixes for generation.
            output_file: the file where output will go.
        """
        super(TrainRnnModelStage, self).__init__(parent)
        self.input_file = input_file
        self.output_file = output_file

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing RNN model text generation stage")
        self.logger.info("Using tokens from {}".format(self.input_file))
        self.logger.info("-" * 40)

    def run(self):
        """Generate text using pre-trained rnn model.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting text generation...")

        self.logger.info("Loading the model...")

        self.logger.info("Loading prefixes...")
        return True
