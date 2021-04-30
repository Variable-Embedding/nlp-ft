"""Experimenting with imdb and fourier transform
"""
from src.stages.stage_benchmark2embeddings import Benchmark2Embeddings
from src.stages.stage_get_pre_trained_embedding import GetPreTrainedEmbeddingsStage
from src.stages.stage_get_benchmark_corpra import GetBenchmarkCorpra
from src.stages.base_stage import BaseStage
from src.stages.stage_train_rnn_model import TrainRnnModelStage
import logging


class RunFTExperiment(BaseStage):
    """Stage for training rnn model.
    """
    name = "run_ft_experiment"
    logger = logging.getLogger("pipeline").getChild(name)

    def __init__(self
                 , corpus_type
                 , embedding_type
                 , parent=None
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
                 , min_freq=1
                 ):
        """Initialization for experimental stages.

        Args:
            corpus_type: required string, name of a benchmark corpus like "wikitext2"
            embedding_type: required, string, name of a pre-trained embedding like "glove.6B.300d"
            parent: optional string, default to meta description of experiment
            batch_size: optional, integer, default to 128
            max_init_param: optional, float, default to 0.05
            max_norm: optional, integer, default to 5
            number_of_layers: optional, integer, default to 2
            sequence_length: optional, integer, default to 30
            sequence_step_size: optional, integer, default to 10
            dropout_probability: optional, float, default to .1
            lstm_configuration: optional, default to 'default'
            device: optional, string, default to "gpu" if exists
            model_type: optional, string, default to
            learning_rate_decay: optional, float, default to 0.05
            learning_rate: optional, float, default to 1.0
            number_of_epochs: optional, integer, default to 2
            lstm_configs: optional, a list of strings, default to ["default"]
            min_freq: optional, integer, default to 1 for filtering out infrequent words to unk token
        """
        super().__init__(parent)
        self.corpus_type = corpus_type
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.min_freq = min_freq
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
        self.logger.info("Executing Experiments with Embeddings of {}".format(self.embedding_type))
        self.logger.info("Experimenting with model Architectures of {}".format(self.model_type))
        self.logger.info("Experimenting with tokens from {}".format(self.corpus_type))
        self.logger.info("-" * 40)

    def run(self):
        """
        """
        # TODO: Set up a way to consolidate and produce a final report of all experiments.
        for corpus in self.corpus_type:

            for embedding in self.embedding_type:
                benchmark = GetBenchmarkCorpra(corpus_type=corpus, parent=self.parent)
                benchmark.run()
                corpra_object = benchmark.corpra

                pre_trained_embedding = GetPreTrainedEmbeddingsStage(embedding_type=embedding, parent=self.parent)
                pre_trained_embedding.run()

                data = Benchmark2Embeddings(embedding_type=embedding
                                            , corpra_object=corpra_object
                                            , min_freq=self.min_freq
                                            , parent=self.parent
                                            , corpus_type=corpus)
                data.run()
                token_sample_nums = data.corpra_numeric['train'][0][1].tolist()
                token_sample_txts = [data.vocab.itos[i] for i in token_sample_nums][0]
                token_sample_vectors = [data.vocab.vectors[i] for i in token_sample_nums]
                print(token_sample_nums[:20])
                print(token_sample_txts[:20])
                print(len(token_sample_nums))
                print(len(token_sample_vectors))
                print(token_sample_vectors[0])

                # test to see what ft does to embedding space

        return True

