"""Base class for stage template
"""
import abc
import logging

class BaseStage(metaclass=abc.ABCMeta):
    """Base class for stage in the pipeline / workflow
    """
    logger = logging.getLogger("pipeline").getChild("base_stage")
    name = "base"

    def __init__(self, parent=None):
        """The init function.

        Args:
            parent: parent stage.
        """
        self.parent = parent

    def execute(self):
        """The function that is called from the outside to execute the stage.

        Returns:
            True if the stage execution succeded, False otherwise.
        """
        self.pre_run()
        if self.run():
            self.post_run()
            return True
        else:
            self.on_failure()
            return False

    def pre_run(self):
        """The function that is executed before the stage is run.
        """

    @abc.abstractmethod
    def run(self):
        """Runs the stage.

        Returns:
            True if the stage execution succeded, False otherwise.
        """

    def post_run(self):
        """The function that is executed after the stage is run successfully.
        """
        self.logger.info("-" * 40)
        self.logger.info("=" * 40)

    def on_failure(self):
        """The function that is executed after the stage run failed.
        """
