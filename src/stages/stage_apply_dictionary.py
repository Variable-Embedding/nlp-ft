"""Stage for performing frequency filtering on corpora.
"""
from src.stages.base_stage import BaseStage
from src.util import constants
from src.util.file import save_integer_tokens_to_file, get_tokens_from_file
from src.util.dictionary import apply_dictionary_to_tokens, change_to_unk, dictionary_file_path

from os.path import join
from collections import Counter

import json
import logging


class ApplyDictionaryStage(BaseStage):
    """Stage for applying dictionary on a text file.
    """
    name = "apply_dictionary"
    logger = logging.getLogger("pipeline").getChild("apply_dictionary_stage")

    def __init__(self, parent=None, input_file=None, output_file=None):
        """Initialization for apply dictionary stage.

        Args:
            parent: the parent stage.
            corpus_file: file to apply the dictionary on.
        """
        super(ApplyDictionaryStage, self).__init__(parent)
        self.input_file = input_file
        self.output_file = output_file

    def pre_run(self):
        """The function that is executed before the stage is run.

        Args:
            args: command line arguments that are passed to the stage.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing dictionary apply stage.")
        self.logger.info("Target file: {}".format(self.input_file))
        self.logger.info("-" * 40)

    def run(self):
        """Run analysis on the corpus file.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting applying dictionary...")
        file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.input_file))
        tokens = get_tokens_from_file(file_path)

        self.logger.info("Loading dictionary...")
        with open(dictionary_file_path(self.parent.topic)) as file:
            dictionary = json.loads(file.read())

        self.logger.info("Changing unknown tokens to <unk>...")
        count = change_to_unk(dictionary, tokens)
        self.logger.info("Changed {} tokens to <unk>.".format(count))

        self.logger.info("Applying dictionary...")
        mapped_tokens = apply_dictionary_to_tokens(dictionary, tokens)

        self.logger.info("Saving the result...")
        output_file_path = join(constants.TMP_PATH,
                                "{}.{}".format(self.parent.topic, self.output_file))
        save_integer_tokens_to_file(mapped_tokens, output_file_path)
        return True
