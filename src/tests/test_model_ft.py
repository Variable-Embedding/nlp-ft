from unittest import TestCase

from src.stages.stage_get_benchmark_corpra import GetBenchmarkCorpra
from src.stages.stage_get_pre_trained_embedding import GetPreTrainedEmbeddingsStage
from src.stages.stage_benchmark2embeddings import Benchmark2Embeddings


class TestModel(TestCase):
    def test_forward(self):
        corpus_type = 'wikitext2'
        embedding_type = 'glove.6B.50d'

        benchmark = GetBenchmarkCorpra(corpus_type=corpus_type)
        benchmark.run()
        print(len(benchmark.corpra[2]))

        pre_trained_embedding = GetPreTrainedEmbeddingsStage(embedding_type=embedding_type)
        pre_trained_embedding.run()

        # data = Benchmark2Embeddings(embedding_type=embedding_type, corpus_type=corpus_type)
        # data.run()
