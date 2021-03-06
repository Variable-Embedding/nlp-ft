"""Stage for comparing different rnn trainings.
"""
from src.model.rnn_model import train_model, test_model, Model, complete_sequence
from src.stages.base_stage import BaseStage
from src.util import constants
from src.util.dictionary import dictionary_file_path
from src.util.file import get_integer_tokens_from_file

from os.path import join

import json
import logging
import matplotlib.pyplot as plt
import numpy as np


class RnnTrainingComparisonStage(BaseStage):
    """Stage for comparing rnn training sessions.
    """
    name = "rnn_training_comparison"
    logger = logging.getLogger("pipeline").getChild("rnn_training_comparison")

    def __init__(self, parent=None):
        """Initialization for model training stage.

        Args:
            parent: the parent stage.
        """
        super(RnnTrainingComparisonStage, self).__init__(parent)

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing RNN training comparison stage")
        self.logger.info("-" * 40)

    def run(self):
        """Compare the training sessions.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        plt.figure()
        for lstm_config in ["default", "var-emb", "res-var-emb"]:
            file_path = join(constants.DATA_PATH, "{}.{}.losses.csv".format(self.parent.topic,
                                                                            lstm_config))
            train_valid_losses = np.genfromtxt(file_path, delimiter=", ", skip_header=1)
            train_losses = train_valid_losses[:][0]
            valid_losses = train_valid_losses[:][1]

            plt.plot(valid_losses, label="{} validation perplexity".format(lstm_config))
            plt.plot(train_losses, label="{} training perplexity".format(lstm_config))

        plt.xlabel("epoch")
        plt.ylabel("perplexity")
        plt.yscale("log")
        plt.legend()
        plt.savefig(join(constants.DATA_PATH, "{}.training_comparison.png".format(
            self.parent.topic)))
        return True
