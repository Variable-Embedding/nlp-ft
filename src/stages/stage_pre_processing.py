"""Stage for pre-processing text.
"""
from src.stages.base_stage import BaseStage
from src.util import constants

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from os.path import join

import logging
import nltk
import re


nltk.download('punkt')
nltk.download('wordnet')

class PreProcessingStage(BaseStage):
    """Stage for pre processing text.
    """
    name = "pre_process"
    logger = logging.getLogger("pipeline").getChild("pre_process")

    def __init__(self, parent=None, input_file=None, output_file=None):
        """Initialization for pre-processing stage.

        Args:
            parent: the parent stage.
            input_file: the input file.
            output_file: the output file
        """
        super(PreProcessingStage, self).__init__(parent)
        self.input_file = input_file
        self.output_file = output_file

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing text pre-processing stage")
        self.logger.info("Target file: {}".format(self.input_file))
        self.logger.info("-" * 40)

    def run(self):
        """Pre-process text.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.logger.info("Starting text pre-processing...")
        input_file_path = join(constants.TMP_PATH,
                               "{}.{}".format(self.parent.topic, self.input_file))
        output_file_path = join(constants.TMP_PATH,
                                "{}.{}".format(self.parent.topic, self.output_file))

        with open(input_file_path, "r") as file:
            text = file.read()

        lemmatizer = WordNetLemmatizer()
        articles = text.split("<article_end>")
        for i in range(len(articles)):
            article = articles[i]
            # Removing article start tokens
            article = re.sub('<article_start>\n', '<article_start> ', article)
            article = re.sub('<article_start>', '', article)
            # Adding paragraph end tokens
            article = re.sub('\n\s*', ' new_line_character ', article)
            article = re.sub('<unk>', ' unkown_token ', article)

            #tokens = word_tokenize(article)
            #for j in range(len(tokens)):
            #    token = tokens[j]
            #    token = token.lower()
            #    token = token.encode("ascii", "ignore")
            #    token = token.decode()
            #    token = lemmatizer.lemmatize(token)
            #    tokens[j] = token
            #article = " ".join(tokens)

            article = re.sub('new_line_character', ' </s> ', article)
            article = re.sub('unkown_token', '<unk>', article)
            articles[i] = "<article_start> {} <article_end>".format(article)
        text = " ".join(articles)

        #self.logger.info("Changing years to <year>")
        #text = re.sub(' \d{4}(\-\d+|s)?', ' <year> ', text)

        #self.logger.info("Changing numbers to <number>")
        #text = re.sub(' \d[\d\.,%]*(st|nd|rd|th| %)?', ' <number>', text)
        #text = re.sub('<number>\-[\d\.\\,%]+', ' <number> ', text)

        self.logger.info("Section title formatting")
        text = re.sub('==+(.*?)==+', '<section_title_start> \\1 <section_title_end>', text)

        self.logger.info("Removing extra white-spaces")
        text = re.sub('\n\s*', ' </s> ', text)
        text = re.sub('\s\s+', ' ', text)

        with open(output_file_path, "w") as file:
            file.write(text)
            num_tokens = len(text.split(" "))
            self.logger.info("Saved the pre-processed text. Contains ~ {} tokens".format(
                num_tokens))
        return True
