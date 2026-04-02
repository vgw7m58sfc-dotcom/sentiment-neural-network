# Sentiment Analysis 

## Overview
This project builds a neural network to perform sentiment analysis on text data using the IMDB dataset.

It compares standard activation functions with adapted versions.

## Features
- Neural network implemented using TensorFlow/Keras
- IMDB dataset
- Comparison of:
  - ReLU
  - Sigmoid
  - Adapted ReLU
  - Adapted Sigmoid
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Early stopping to prevent overfitting
- Automatic generation of performance plots

## Installation

1. Clone the repository:

git clone https://github.com/vgw7m58sfc-dotcom/sentiment-neural-network.git

cd sentiment-neural-network

2. Install packages:

pip install -r packagesrequired.txt


## Running the code

Run the main script:

python sentiment_analysis.py


## Output
The script will:
- Train four neural networks using different activation functions
- Print performance metrics in the terminal
- Generate and save plots:
  - training_loss.png
  - validation_loss.png
  - accuracy.png
