"""Stage for retrieving and processing benchmark corpra with torchtext api.
"""
from src.stages.base_stage import BaseStage
from src.util import constants
import logging
import os
import time
from functools import partial
# language modeling corpra
import torchtext
from torchtext.datasets.wikitext2 import WikiText2
from torchtext.datasets.wikitext103 import WikiText103
from torchtext.datasets.penntreebank import PennTreebank

# for classification tasks
from torchtext.datasets.ag_news import AG_NEWS
from torchtext.datasets.amazonreviewfull import AmazonReviewFull
from torchtext.datasets.amazonreviewpolarity import AmazonReviewPolarity
from torchtext.datasets.dbpedia import DBpedia
from torchtext.datasets.yelpreviewfull import YelpReviewFull
from torchtext.datasets.yelpreviewpolarity import YelpReviewPolarity
from torchtext.datasets.yahooanswers import YahooAnswers
from torchtext.datasets.imdb import IMDB


def get_benchmark(corpus_type, benchmark, logger):
    """A helper function to download benchmark corpra with torchtext api.

    :param corpus_type: a string, representing the target corpus
    :param benchmark: a dictionary of torchtext partial functions for downloading corpra, keyed by string
    :param logger: the logging object.
    """

    if any([corpus_type in i for i in benchmark.keys()]):
        logger.info(f'Retrieving benchmark corpra for {corpus_type}')
        if isinstance(benchmark[corpus_type], list):
            for b in benchmark[corpus_type]:
                logger.info(f'Getting {corpus_type}')
                b()
                # wait for a brief period to avoid server abuse
                time.sleep(3)
        else:
            return benchmark[corpus_type]()

    else:
        logger.info(f'Selected corpus name of "{corpus_type}" not found in list of known benchmarks, try any of \n {list(benchmark.keys())}')


def check_data_folder_path():
    """A helper function to check for a directory and make it if not exists.
    :return: an os path object for the current path
    """

    if not os.path.exists(constants.DATA_PATH):
        os.makedirs(constants.DATA_PATH)
    return constants.DATA_PATH


def frozen_benchmarks():
    """A helper function to freeze available torchtext api functions for getting benchmarks
    :return: benchmark, a dictionary of torchtext functions, keyed by a simple name.
    """
    root = check_data_folder_path()
    # language modeling tasks
    wikitext2 = partial(WikiText2, root=root)
    wikitext103 = partial(WikiText103, root=root)
    # ptb is special case, needs its own folder b/c it is not zipped data
    ptb_root = os.sep.join([root, 'penntreebank'])
    penntreebank = partial(PennTreebank, root=ptb_root)
    language_modeling_tasks = [wikitext2, wikitext103, penntreebank]

    # classification tasks
    ag_news = partial(AG_NEWS, root=root)
    amazon_review_full = partial(AmazonReviewFull, root=root)
    amazon_review_polarity = partial(AmazonReviewPolarity, root=root)
    db_pedia = partial(DBpedia, root=root)
    yelp_review_full = partial(YelpReviewFull, root=root)
    yelp_review_polarity = partial(YelpReviewPolarity, root=root)
    yahoo_answers = partial(YahooAnswers, root=root)
    imdb = partial(IMDB, root=root)
    classification_tasks = [ag_news, amazon_review_polarity, amazon_review_full, db_pedia, yelp_review_polarity, yelp_review_full, yahoo_answers, imdb]
    all_benchmarks = language_modeling_tasks + classification_tasks

    benchmark = {"everything": all_benchmarks,
                 "language_modeling_tasks": language_modeling_tasks,
                 "classification_tasks": classification_tasks,
                 "wikitext2": wikitext2,
                 "wikitext103": wikitext103,
                 "penntreebank": penntreebank,
                 "ag_news": ag_news,
                 "amazon_review_polarity": amazon_review_polarity,
                 "amazon_review_full": amazon_review_full,
                 "db_pedia": db_pedia,
                 "yelp_review_polarity": yelp_review_polarity,
                 "yelp_review_full": yelp_review_full,
                 "yahoo_answers": yahoo_answers,
                 "imdb": imdb}

    return benchmark


class GetBenchmarkCorpra(BaseStage):
    """Stage for retrieving benchmark corpus with torchtext api.
    """
    name = "get_benchmark_corpra"
    logger = logging.getLogger("pipeline").getChild("get_benchmark_corpra_stage.")

    def __init__(self, parent=None, corpus_type=None):
        """Initialization for Get Benchmark Corpra Stage.
        :param corpus_type: a string, representation of the target corpus type or modeling task. Default to "everything".

        list of possible options for corpus_type:

        'language_modeling_tasks' -> downloads three corpra,
        'classification_tasks' -> downloads eight corpra,
        'wikitext2', 'wikitext103', 'penntreebank',
        'ag_news',
        'amazon_review_polarity', 'amazon_review_full',
        'db_pedia'
        'yelp_review_polarity', 'yelp_review_full',
        'yahoo_answers',
        'imdb',
        'everything' -> downloads everything available

        """
        super().__init__(parent)

        self.corpus_type = 'everything' if corpus_type is None else corpus_type
        self.corpra = None

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing get torchtext corpus stage.")
        self.logger.info("-" * 40)

    def run(self):
        """Retrieves benchmark texts with torchtext api.

        :return: True if the stage execution succeeded, False otherwise.
        """
        benchmark = frozen_benchmarks()
        self.corpra = get_benchmark(corpus_type=self.corpus_type
                                    , benchmark=benchmark
                                    , logger=self.logger
                                    )

        return True