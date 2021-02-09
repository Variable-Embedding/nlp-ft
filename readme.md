# NLP pipeline

This is an NLP pipeline used for MSAI 337 course.

## Overview

This is a natural-language processing pipeline. Currently it supports these stages:

1. Scrape articles from the Wikipedia.
2. Clean the scraped text.
3. Split the text into training / testing / validation files.
4. Pre-process text file.
5. Build a dictionary from a text file.
6. Apply a dictionary to convert text file to a list of integer tokens.
7. Train RNN model on the list of tokens.

## Executing pipeline / workflow

Edit the pipeline config files to run the stages that you want, and run the following command:
```
make run
```

To only scrap the wikipedia:
```
make wikipedia-scraping
```

To run the RNN model:
```
make model-our-data
```

To clean the directory:
```
make clean
```

## Logging

This project uses logging library. The workflow generates log files that can be found in logs folder. Use logger.info / debug / error / warning instead of print for proper logging when creating new stages.
