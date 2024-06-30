# [IMDB DATASET](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) Sentiment Analysis Model

## Project Overview

This project implements a sentiment analysis model using the IMDB Dataset of
50,000 movie reviews. The model aims to classify movie reviews as positive or
negative based on their content.

## Tokenization

A custom Byte-Pair Encoding (BPE) tokenizer located in the `tokenizer` directory
has been implemented for this project, enabling reduced model training time and
optimized vocabulary size. BPE was chosen for its ability to handle
out-of-vocabulary words effectively, ensuring no unknown tokens in this dataset.

## Helper Functions

Utility functions are defined in the `util` directory to to avoid lengthy code
for trivial metric calculation.

## Model Architecture

While modern NLP has been revolutionized by transformers the introduction of
transformers in Google's `"Attention is All You Need"` paper, this project
focuses on fundamental architectures. I implemented a LSTM network, as a
stepping stone towards understanding more complex models like transformers.
