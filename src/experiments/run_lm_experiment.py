"""Runner file for language modeling experiments.
"""
from src.stages.stage_benchmark2embeddings import Benchmark2Embeddings
from src.stages.stage_get_pre_trained_embedding import GetPreTrainedEmbeddingsStage
from src.stages.stage_get_benchmark_corpra import GetBenchmarkCorpra
from src.stages.base_stage import BaseStage
from src.stages.stage_train_rnn_model import TrainRnnModelStage
import logging


class RunLMExperiment(BaseStage):
    """Stage for training rnn model.
    """
    name = "run_lm_experiment"
    logger = logging.getLogger("pipeline").getChild("run_lm_experiment")

    def __init__(self
                 , parent=None
                 , corpus_type=None
                 , embedding_type=None
                 , batch_size=128
                 , max_init_param=0.05
                 , max_norm=5
                 , number_of_layers=2
                 , sequence_length=30
                 , sequence_step_size=10
                 , dropout_probability=0.1
                 , lstm_configuration='default'
                 , device="gpu"
                 , model_type="lstm"
                 , learning_rate_decay=0.85
                 , learning_rate=1
                 , number_of_epochs=1
                 , lstm_configs=None
                 ):
        """Initialization for experimental stages.
        """
        super(RunLMExperiment, self).__init__(parent)
        self.corpus_type = corpus_type
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.model_config = {
            'batch_size': batch_size
            , 'max_init_param': max_init_param
            , 'max_norm': max_norm
            , 'number_of_layers': number_of_layers
            , 'sequence_length': sequence_length
            , 'sequence_step_size': sequence_step_size
            , 'dropout_probability': dropout_probability
            , 'lstm_configuration': lstm_configuration
            , 'device': device
            , 'lstm_configs': lstm_configs
        }
        self.train_config = {
            'learning_rate_decay': learning_rate_decay
            , 'learning_rate': learning_rate
            , 'number_of_epochs': number_of_epochs
        }

    def pre_run(self):
        """The function that is executed before the stage is run.
        """
        self.logger.info("=" * 40)
        self.logger.info("Executing Experiments for {}".format(list(zip(self.corpus_type, self.embedding_type))))
        self.logger.info("Using tokens from {}".format(self.corpus_type))
        self.logger.info("-" * 40)

    def run(self):
        """
        """
        for corpus in self.corpus_type:
            benchmark = GetBenchmarkCorpra(corpus_type=corpus)
            benchmark.run()
            corpra_object = benchmark.corpra

            for embedding in self.embedding_type:
                pre_trained_embedding = GetPreTrainedEmbeddingsStage(embedding_type=embedding)
                pre_trained_embedding.run()

                data = Benchmark2Embeddings(embedding_type=embedding, corpra_object=corpra_object)
                data.run()

                for model_type in self.model_type:
                    self.model_config.update({'model_type': model_type})
                    trainer = TrainRnnModelStage(corpus_type=corpus
                                                 , train_file=data.corpra_numeric['train']
                                                 , valid_file=data.corpra_numeric['valid']
                                                 , test_file=data.corpra_numeric['test']
                                                 , model_config=self.model_config
                                                 , train_config=self.train_config
                                                 , vectors=data.vocab.vectors
                                                 , dictionary=data.vocab.stoi
                                                 )
                    trainer.run()


        return True

