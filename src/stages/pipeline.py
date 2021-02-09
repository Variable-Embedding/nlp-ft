"""Data pipeline / workflow.
"""
from src.stages.base_stage import BaseStage
from src.util import constants

from os.path import join

import logging


class Pipeline(BaseStage):
    """A class for the whole data pipeline / workflow
    """
    logger = logging.getLogger("pipeline")

    def __init__(self, parent=None, stages=None, topic="default"):
        """Init function for the pipeline.

        Args:
            parent: parent stage.
            stages: a list of stages.
        """
        super(Pipeline, self).__init__(parent)
        self.stages = stages
        for stage in self.stages:
            stage.parent = self
        self.topic = topic

    def pre_run(self):
        """The function that is executed before the pipeline / workflow is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("-" * 40)
        self.logger.info("Starting data pipeline / workflow")
        self.logger.info("-" * 40)
        self.logger.info("=" * 40)

    def run(self):
        """Run the stages specified in the stage argument.

        Returns:
            True if the workflow / pipeline execution succeded, False otherwise.
        """
        for stage in self.stages:
            self.logger.info("Executing stage '{}'".format(stage.name))
            if not stage.execute():
                return False

        return True
