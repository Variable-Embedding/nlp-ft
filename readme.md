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
git clone https://github.com/iryzhkov/nlp-pipeline
```

* The secondary branch ("nn_embedding") comprises work related to experiments with variable emebdding. To switch branches:

```terminal
git checkout nn_embedding
```

2. After cloning, selecting a branch, and an IDLE of your choosing, from terminal, run the command make install to load all dependencies:

```terminal
make install
```

3. Alternatively you can run this project from Google Colab at the following URL [https://colab.research.google.com/drive/13EOfZHtKrgWWwHnF8rGUOewVtwND9Ns-?usp=sharing](https://colab.research.google.com/drive/13EOfZHtKrgWWwHnF8rGUOewVtwND9Ns-?usp=sharing). Running the COLAB notebook will clone the repository, install depenencides and run the experiement through your browser.

## Executing pipeline / workflow

Edit the pipeline config files to run the stages that you want, and run the following command:

```terminal
make run
```

To only scrap the wikipedia:

```terminal
make wikipedia-scraping
```

To run the RNN model:
```
make model-our-data
```

To clean the directory:

```terminal
make clean
```

## Logging

This project uses logging library. The workflow generates log files that can be found in logs folder. Use logger.info / debug / error / warning instead of print for proper logging when creating new stages.
