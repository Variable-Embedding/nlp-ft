# NLP pipeline

This is a natural language processing (NLP) pipeline built for research during [MSAI 337, Introduction to Natural Language Processing](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-337.html).

## Overview

NLP pipeline. Currently it supports these stages:

1. Scrape articles from Wikipedia.
2. Cleans scraped text.
3. Splits text into training / testing / validation files.
4. Pre-process text file.
5. Build a dictionary from a text file.
6. Apply a dictionary to convert text file to a list of integer tokens.
7. Train RNN model on the list of tokens.

## Getting Started

1. There are currently two branches of interest in the respository. Before cloning, consider which set of code you are interested in.

* The main branch ("master") comprises work related to MSAI 337 class deliverbale #1 and #2. Cloning this respository will default to the main branch.

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

* Get pre-trained glove embeddings. By default, the get_pre_trained_embeddings.yaml file is set to the smallest embedding file for glove.6B.100d.
```terminal
make get-glove
```

* See [torchtext-glove](https://torchtext.readthedocs.io/en/latest/vocab.html) documents on pretrained_aliases for a full listing of available downloads.

## Logging

This project uses logging library. The workflow generates log files that can be found in logs folder. Use logger.info / debug / error / warning instead of print for proper logging when creating new stages.
