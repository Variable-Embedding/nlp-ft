"""Stage running training iterations.
"""
from src.model.model_base import Model
from src.model.model_tools import train_model, test_model
from src.stages.base_stage import BaseStage
from src.util import constants

from os.path import join

import logging
import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainRnnModelStage(BaseStage):
    """Stage for training rnn model.
    """
    name = "train_rnn_model"
    logger = logging.getLogger("pipeline").getChild("train_rnn_model")

    def __init__(self
                 , parent=None
                 , corpus_type=None
                 , vectors=None
                 , train_file=None
                 , test_file=None
                 , valid_file=None
                 , dictionary=None
                 , model_config=None
                 , train_config=None
                 ):
        """Initialization for model training stage.
        """
        super(TrainRnnModelStage, self).__init__(parent)
        self.train_file = train_file
        self.test_file = test_file
        self.valid_file = valid_file
        self.dictionary = dictionary
        self.vectors = vectors
        self.corpus_type = corpus_type

        self.model_config = model_config
        self.train_config = train_config
        self.lstm_configs = model_config['lstm_configs'] if model_config['lstm_configs'] is not None else model_config['lstm_configiratopm']
        self.dictionary = dictionary

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing RNN model training stage")
        self.logger.info("Using tokens from {}".format(self.train_file))
        self.logger.info("-" * 40)

    def run(self):
        """Train the model.

        Returns:
            True if the stage execution worked, False otherwise.
        """

        train_tokens = self.train_file
        self.logger.info("Loaded {} tokens.".format(len(train_tokens)))

        if self.test_file is not None:
            test_tokens = self.test_file
            self.logger.info("Loaded {} tokens.".format(len(test_tokens)))
        else:
            test_tokens = None

        if self.valid_file is not None:
            valid_tokens = self.valid_file
            self.logger.info("Loaded {} tokens.".format(len(valid_tokens)))
        else:
            valid_tokens = None

        self.logger.info("Loading dictionary...")

        self.logger.info("Dictionary contains {} tokens.".format(len(self.dictionary)))

        for lstm_config in self.lstm_configs:
            self.parent.topic = self.lstm_config
            model = Model(dictionary_size=len(self.dictionary)
                          , embedding_vectors=self.vectors
                          , embedding_size=self.vectors.size()[1]
                          , **self.model_config)
            self.logger.info("Starting model training with lstm configuration {} ...".format(
                lstm_config))
            train_losses, valid_losses = train_model(model=model
                                                     , train_tokens=train_tokens
                                                     , valid_tokens=valid_tokens
                                                     , logger=self.logger
                                                     , **self.train_config)
            self.logger.info("Finished model training.")
            self.logger.info("Saving the model...")
            file_path = join(constants.DATA_PATH, "{}.{}.model.pkl".format(self.parent,
                                                                           self.parent.topic))
            torch.save(model, file_path)

            self.logger.info("Saving training and validation losses to csv...")
            train_valid_losses = np.column_stack((train_losses, valid_losses[1:]))
            file_path = join(constants.DATA_PATH, "{}.{}.losses.csv".format(self.parent,
                                                                            self.parent.topic))
            np.savetxt(file_path, train_valid_losses, delimiter=", ", header="train, valid")

            self.logger.info("Performing model evaluation...")
            self.logger.info("Test perplexity score: {:.1f}".format(test_model(model, test_tokens)))
            #self.logger.info("Train perplexity score: {:.1f}".format(test_model(model, train_tokens)))
            self.logger.info("Valid perplexity score: {:.1f}".format(valid_losses[-1]))
            plt.figure()
            plt.plot(valid_losses[1:], label="validation perplexity")
            plt.plot(train_losses, label="training perplexity")
            plt.xlabel("epoch")
            plt.ylabel("perplexity")
            plt.yscale("log")
            plt.legend()
            plt.savefig(join(constants.DATA_PATH, "{}.{}.preplexity.png".format(self.parent,
                                                                                self.parent.topic)))
        return True
