# NLP pipeline

This is a natural language processing (NLP) pipeline built for research during [MSAI 337, Introduction to Natural Language Processing](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-337.html).

## Overview

NLP pipeline for Spring 2021.

1. Get pre-trained GloVe embeddings.

## Getting Started

```terminal
git clone https://github.com/Variable-Embedding/nlp-ft
```

* To run from Google Colab, make a copy of the following colab notebook: [https://colab.research.google.com/drive/1WI6atgi6TW8wikipHfj5adOAJ_oZrwy6?usp=sharing](https://colab.research.google.com/drive/1WI6atgi6TW8wikipHfj5adOAJ_oZrwy6?usp=sharing)

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

3. Run the experiment pipeline for language models ("lm"). 
```terminal
make lm-experiement
```

4. Options to get and prep necessary data for nlp pipeline.

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

* The LM experiment can be controlled with run_lm_experiment_pipeline.yaml:
```terminal

# The following configuration will run experiements for three copra, 
# each with the three version of glove embeddings, 
# each with the given training and model params, 
# run each with the lstm architecture, 
# and run two variations of the lstm. 

# in effect, we are running the experiment n times for each corpus with m configurations.

stages:
- name: run_lm_experiment
  corpus_type:
  - wikitext2
  - penntreebank
  embedding_type:
  - glove.6B.50d
  - glove.840B.300d
  model_type:
    - lstm
    - lstm
  lstm_configs:
    - default
    - res-ff-emb
  batch_size: 1024
  max_init_param: 0.05
  max_norm: 5
  number_of_layers: 2
  sequence_length: 30
  sequence_step_size: 10
  dropout_probability: 0.1
  device: gpu
  learning_rate_decay: 0.85
  learning_rate: 1
  number_of_epochs: 2
  min_freq: 5
```

5. Optional: Setup and configure pypy3 (THIS IS EXPERIMENTAL - NOT WORKING RIGHT NOW).
* Create a new virtual environment or conda environment, then activate it. 
* For MacOS users, brew install pypy3
```terminal
brew install pypy3
```
* Configure pypy3 and install dependencies:
```terminal
make install-pypy
```

* For help on pypy3 and brew, see [pypy3 docs](https://doc.pypy.org/en/latest/install.html) and [brew formula](https://formulae.brew.sh/formula/pypy3).

* See [torchtext-glove](https://torchtext.readthedocs.io/en/latest/vocab.html) documents on pretrained_aliases for a full listing of available downloads.

## Logging

This project uses logging library. The workflow generates log files that can be found in logs folder. Use logger.info / debug / error / warning instead of print for proper logging when creating new stages.
