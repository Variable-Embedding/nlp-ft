"""Stage for creating a dictionary for a corpora.
"""
from src.stages.base_stage import BaseStage
from src.util import constants
from src.util.file import get_tokens_from_file
from src.util.dictionary import get_dictionary_from_tokens, dictionary_file_path

from os.path import join
from collections import Counter

import json
import logging


class DictionaryCreationStage(BaseStage):
    """Stage for creating a dictionary.
    """
    name = "dictionary_creation"
    logger = logging.getLogger("pipeline").getChild("dictionary_creation_stage")

    def __init__(self, parent=None, input_file=None, frequency_threshold=0):
        """Initialization for dictionary creation stage.

        Args:
            parent: The parent stage.
            corpus_file: corpus file to create dictionary from.
            frequency_threshold: minimum number of times a token has to appear.
        """
        super(DictionaryCreationStage, self).__init__(parent)
        self.frequency_threshold = frequency_threshold
        self.input_file = input_file

    def pre_run(self):
        """The function that is executed before the stage is run.

        Args:
            args: command line arguments that are passed to the stage.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing dictionary creation stage.")
        self.logger.info("Frequency Threshold: {}".format(self.frequency_threshold))
        self.logger.info("Target file: {}".format(self.input_file))
        self.logger.info("-" * 40)

    def run(self):
        """Run analysis on the corpus file.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Loading tokens from corpus...")
        file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.input_file))
        tokens = get_tokens_from_file(file_path)

        self.logger.info("Generating dictionary from loaded tokens...")
        dictionary = get_dictionary_from_tokens(tokens, self.frequency_threshold)

        self.logger.info("Dictionary contains {} tokens".format(len(dictionary)))
        self.logger.info("Saving dictionary...")
        with open(dictionary_file_path(self.parent.topic), 'w') as file:
            file.write(json.dumps(dictionary))
        return True
