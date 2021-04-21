"""Stage for pre-processing text.
"""
from src.stages.stage_benchmark2embeddings import Benchmark2Embeddings
from src.model.rnn_model import train_model, test_model, Model
from src.stages.base_stage import BaseStage
from src.util import constants
from src.util.dictionary import dictionary_file_path
from src.util.file import get_integer_tokens_from_file

from os.path import join
import os

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


class TrainRnnModelStage(BaseStage):
    """Stage for training rnn model.
    """
    name = "train_rnn_model"
    logger = logging.getLogger("pipeline").getChild("train_rnn_model")

    def __init__(self
                 , parent=None
                 , corpus_type=None
                 , embedding_type=None
                 , train_file=None
                 , test_file=None
                 , valid_file=None
                 , model_config_file=None
                 , training_config_file=None
                 , lstm_configs=None
                 , dictionary=None
                 ):
        """Initialization for model training stage.
        """
        super(TrainRnnModelStage, self).__init__(parent)

        data = Benchmark2Embeddings(embedding_type=embedding_type, corpus_type=corpus_type)
        data.run()

        corpra = make_benchmark_corpra(data.corpra_cache, data.vocab, data.tokenizer)

        self.train_file = corpra['train'] if corpra else train_file
        self.test_file = corpra['test'] if corpra else test_file
        self.valid_file = corpra['valid'] if corpra else valid_file
        self.dictionary = data.vocab.stoi if data else dictionary
        self.vectors = data.vocab.vectors

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

        self.logger.info("Loading model and training configurations...")
        with open(self.model_config_filepath, "r") as file:
            model_config = yaml.safe_load(file)
        with open(self.training_config_filepath, "r") as file:
            training_config = yaml.safe_load(file)

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
            model_config["lstm_configuration"] = lstm_config
            model = Model(dictionary_size=len(self.dictionary)
                          , embedding_vectors=self.vectors
                          , embedding_size=self.vectors.size()[1]
                          , **model_config)
            self.logger.info("Starting model training with lstm configuration {} ...".format(
                lstm_config))
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


def make_benchmark_corpra(cache_paths, vocab, tokenizer):
    """Leveraging torchtext functions.
    """

    logging.info('Starting make_torch_corpra()')

    corpra = {}

    for cache_path in cache_paths:

        key = 'train' if '.train.' in cache_path else 'test' if '.test.' in cache_path else 'valid'

        f = open(cache_path, 'r')

        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

        corpus = []

        for line in f:
            c = text_pipeline(line)
            corpus.extend(c)

        corpus = torch.tensor(corpus, dtype=torch.long)
        corpra.update({key: corpus})

        f.close()

    return corpra
