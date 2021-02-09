"""Stage for scrapping the text data from the internet.
"""
from src.stages.base_stage import BaseStage
from src.util import constants

from os.path import join

import logging
import math
import random
import re


class CorpusSplitStage(BaseStage):
    """Stage for splitting corpus.
    """
    name = "corpus_split"
    logger = logging.getLogger("pipeline").getChild("corpus_split_stage")

    def __init__(self, parent=None, splits=None):
        """Initializer for corpus split stage.

        Args:
            parent: the parent stage
            splits: configuration for the splitting.
        """
        super(CorpusSplitStage, self).__init__(parent)
        self.splits = splits

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing corpus splitting stage")
        self.logger.info("-" * 40)

    def run(self):
        """Splits the corpus into specified corpora.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting corpus splitting...")
        total_proportions = sum([s["proportion"] for s in self.splits])
        input_file_path = join(constants.TMP_PATH, "{}.clean.txt".format(self.parent.topic))

        with open(input_file_path, "r") as file:
            text = file.read()

        text = re.sub("(\n)?<article_start>", "", text)
        articles = text.split("<article_end>")
        articles = ["<article_start>{}<article_end>".format(a) for a in articles]
        random.shuffle(articles)
        num_articles_for_split = [math.floor(len(articles) * (s["proportion"] / total_proportions))
                                  for s in self.splits]
        diff = len(articles) - sum(num_articles_for_split)
        for i in range(len(self.splits)):
            num_articles_for_split[i] += diff // len(self.splits)
            if i < diff % len(self.splits):
                num_articles_for_split[i] += 1

        prev_index = 0
        for i in range(len(self.splits)):
            curr_index = prev_index + num_articles_for_split[i]
            text = " ".join(articles[prev_index:curr_index])
            prev_index = curr_index

            num_tokens = len(text.split(" "))
            self.logger.info("Corpus {} contains ~ {} tokens".format(self.splits[i]["name"],
                                                                     num_tokens))

            output_file_path = join(constants.TMP_PATH,
                                    "{}.{}.txt".format(self.parent.topic, self.splits[i]["name"]))
            with open(output_file_path, "w") as file:
                file.write(text)
        return True
