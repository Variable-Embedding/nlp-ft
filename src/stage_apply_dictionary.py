"""Stage for performing frequency filtering on corpora.
"""
from base_stage import BaseStage
from configuration import run_configuration
from data_functions import save_tokens_to_file, get_tokens_from_file

import constants

from os.path import join
from collections import Counter

import json
import logging

def apply_dictionary(tokens, dictionary):
    """Applies dictionary to tokens.

    Args:
        tokens: a list of tokens.
        dictionary: token to number mapping.

    Returns:
        A list of numbers generated.
    """
    return [dictionary[t][0] for t in tokens]

class ApplyDictionaryStage(BaseStage):
    """Stage for applying dictionary on a text file.
    """
    name = "apply_dictionary"
    logger = logging.getLogger("pipeline").getChild("apply_dictionary_stage")

    def __init__(self, parent=None, corpus_file=None):
        """Initialization for apply dictionary stage.

        Args:
            parent: the parent stage.
            corpus_file: file to apply the dictionary on.
        """
        super(ApplyDictionaryStage, self).__init__(parent)
        self.corpus_file = corpus_file

    def pre_run(self):
        """The function that is executed before the stage is run.

        Args:
            args: command line arguments that are passed to the stage.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing dictionary apply stage.")
        self.logger.info("Target file: {}".format(self.corpus_file))
        self.logger.info("-" * 40)

    def run(self):
        """Run analysis on the corpus file.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting applying dictionary...")
        file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.corpus_file))
        tokens = get_tokens_from_file(file_path)

        self.logger.info("Loading dictionary...")
        dictionary_file_path = join(constants.DATA_PATH,
                                    "{}.dictionary.json".format(self.parent.topic))
        with open(dictionary_file_path) as file:
            dictionary = json.loads(file.read())

        self.logger.info("Changing unknown tokens to <<unk>>...")
        count = 0
        for i, token in enumerate(tokens):
            if not token in dictionary:
                tokens[i] = "<<unk>>"
                count += 1
        self.logger.info("Changed {} tokens".format(count))

        self.logger.info("Applying dictionary...")
        mapped_tokens = apply_dictionary(tokens, dictionary)

        self.logger.info("Saving the result...")
        output_file_path = join(constants.DATA_PATH,
                                "{}.{}".format(self.parent.topic, self.corpus_file))
        save_tokens_to_file(mapped_tokens, output_file_path)
        return True
