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
import matplotlib.pyplot as plt
import numpy as np
import re
import termplotlib as tpl
import torch
import yaml


class TrainRnnModelStage(BaseStage):
    """Stage for training rnn model.
    """
    name = "train_rnn_model"
    logger = logging.getLogger("pipeline").getChild("train_rnn_model")

    def __init__(self, parent=None, train_file=None, test_file=None, valid_file=None,
                 model_config_file=None, training_config_file=None, lstm_configs=None):
        """Initialization for model training stage.

        Args:
            parent: the parent stage.
            train_file: the file with integer tokens used for training.
            test_file: the file with integer tokens used for testing.
            valid_file: the file with integer tokens used for validation.
            model_config_file: the file with model configuration.
            training_config_file: the file with training configuration.
            lstm_config: configuration for lstm model.
        """
        super(TrainRnnModelStage, self).__init__(parent)
        self.train_file = train_file
        self.test_file = test_file
        self.valid_file = valid_file
        self.model_config_filepath = join(constants.CONFIG_PATH, model_config_file)
        self.training_config_filepath = join(constants.CONFIG_PATH, training_config_file)
        self.lstm_configs = ["default"] if lstm_configs is None else lstm_configs

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
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting text pre-processing...")
        self.logger.info("Loading model and training configurations...")
        with open(self.model_config_filepath, "r") as file:
            model_config = yaml.safe_load(file)
        with open(self.training_config_filepath, "r") as file:
            training_config = yaml.safe_load(file)

        self.logger.info("Loading training tokens...")
        file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.train_file))
        train_tokens = list(map(int, get_integer_tokens_from_file(file_path)))
        self.logger.info("Loaded {} tokens.".format(len(train_tokens)))

        if self.test_file:
            self.logger.info("Loading testing tokens...")
            file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.test_file))
            test_tokens = list(map(int, get_integer_tokens_from_file(file_path)))
            self.logger.info("Loaded {} tokens.".format(len(test_tokens)))
        else:
            test_tokens = None

        if self.valid_file:
            self.logger.info("Loading validation tokens...")
            file_path = join(constants.TMP_PATH, "{}.{}".format(self.parent.topic, self.valid_file))
            valid_tokens = list(map(int, get_integer_tokens_from_file(file_path)))
            self.logger.info("Loaded {} tokens.".format(len(valid_tokens)))
        else:
            valid_tokens = None

        self.logger.info("Loading dictionary...")
        with open(dictionary_file_path(self.parent.topic)) as file:
            dictionary = json.loads(file.read())
        self.logger.info("Dictionary contains {} tokens.".format(len(dictionary)))

        for lstm_config in self.lstm_configs:
            model_config["lstm_configuration"] = lstm_config
            model = Model(dictionary_size=len(dictionary), **model_config)
            self.logger.info("Starting model training with lstm configuration {} ...".format(
                lstm_config))

            # Pretrain if model requires
            # NOTE: losses will be overwritten, but only using the last ones make sense anyway
            if lstm_config == "ff-emb-pretrain":
              # Set pretrain step
              self.logger.info("Pretraining step")
              model.set_pretrain()
              # Pretrain
              train_losses, valid_losses = train_model(model=model, train_tokens=train_tokens,
                                                     valid_tokens=valid_tokens, logger=self.logger,
                                                     **training_config)
              # Set back to main training
              self.logger.info("Main training step")
              model.set_main_train()

            
            # Main training
            train_losses, valid_losses = train_model(model=model, train_tokens=train_tokens,
                                                     valid_tokens=valid_tokens, logger=self.logger,
                                                     **training_config)

            self.logger.info("Finished model training.")
            self.logger.info("Saving the model...")
            file_path = join(constants.DATA_PATH, "{}.{}.model.pkl".format(self.parent.topic,
                                                                           lstm_config))
            torch.save(model, file_path)

            self.logger.info("Saving training and validation losses to csv...")
            train_valid_losses = np.column_stack((train_losses, valid_losses[1:]))
            file_path = join(constants.DATA_PATH, "{}.{}.losses.csv".format(self.parent.topic,
                                                                            lstm_config))
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
            plt.savefig(join(constants.DATA_PATH, "{}.{}.preplexity.png".format(self.parent.topic,
                                                                                lstm_config)))
        return True
