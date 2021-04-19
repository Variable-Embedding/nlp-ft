"""Stage for retrieving and processing benchmark corpra.
"""
from src.stages.base_stage import BaseStage
from src.util import constants
import logging

# language modeling corpra
from torchtext.datasets.wikitext2 import WikiText2
from torchtext.datasets.wikitext103 import WikiText103
from torchtext.datasets.penntreebank import PennTreebank

# for classification tasks
from torchtext.datasets.imdb import IMDB

#TODO: build out functions to get all types of text

class GetTorchTextCorpra(BaseStage):
    """Stage for retrieving benchmark corpus with torchtext api.
    """
    name = "get_torchtext_corpus"
    logger = logging.getLogger("pipeline").getChild("get_torchtext_corpus_stage.")

    def __init__(self, parent=None, corpus_type=None):
        """Initialization for Get Torchtext Corpus Benchmark Stage.
        """
        super().__init__(parent)
        self.corpus_type = 'glove.6B.100d' if corpus_type is None else corpus_type

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing get torchtext corpus stage.")
        self.logger.info("-" * 40)

    def run(self):
        """Retrieves benchmark texts with torchtext api.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        print('get torch corpus')

        return True