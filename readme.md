# NLP pipeline

This is a natural language processing (NLP) pipeline built for research during [MSAI 337, Introduction to Natural Language Processing](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-337.html).

## Overview

NLP pipeline for Spring 2021.

1. Get pre-trained GloVe embeddings.

## Getting Started

```terminal
git clone https://github.com/Variable-Embedding/nlp-ft
```

2. After cloning, selecting a branch, and an IDLE of your choosing, from terminal, run the command make install to load all dependencies:

* optional but recommended step: create a new anaconda environment.
```terminal
conda create -n nlp-ft
conda activate nlp-ft
```

* run prep scripts
```terminal
make prep
```

* install dependencies
```terminal
make install
```

3. Get and prep necessary data for nlp pipeline.

* Get pre-trained glove embeddings. By default, the get_pre_trained_embeddings.yaml file is set to return all types of available embeddings "everything".
_Note_: Options: "everything" to get all embedding files or one of: glove.6B, glove.42B, glove.840B, glove.twitter, fasttext.en, fasttext.simple, charngram
```terminal
make embeddings
```


* Get benchmark corpra via the torchtext api such as wikitext 2 and imdb datasets. Provide the name of a corpus such as "imdb", specify a task such as "language_modeling", or get "everything".
_Note_: By default, make benchmark will get everything that is currently available in this script. 
```terminal
make benchmark
```

4. Run the model pipeline. 
```terminal
make model
```

* See [torchtext-glove](https://torchtext.readthedocs.io/en/latest/vocab.html) documents on pretrained_aliases for a full listing of available downloads.

## Logging

This project uses logging library. The workflow generates log files that can be found in logs folder. Use logger.info / debug / error / warning instead of print for proper logging when creating new stages.
