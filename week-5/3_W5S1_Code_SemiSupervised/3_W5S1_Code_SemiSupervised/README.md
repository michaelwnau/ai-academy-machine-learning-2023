# Classic Semi-supervised Learning Algorithms

This repository includes the dataset and solution to the workshop questions.

## Data
The dataset is the austrailian credit card from UCI: https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval)

## Setup
It loads the dataset and applies normalization to the values. Then separates the train and test by 70/30 ratio.
It builds several models for different percentages of labeled data in the semi-supervised learning procedure from 1% to 15%.

## Models
The following models are built and learned:

1. Fully supervised classifiers based on the labeled proportion of data: SVM, LR, DT, and NB
2. Self-learning algorithm with the base classifiers as: SVM, LR, DT, and NB
3. S3VM algorithm imported from [this package](https://github.com/tmadl/semisup-learn)
4. Label Propagation algorithm
