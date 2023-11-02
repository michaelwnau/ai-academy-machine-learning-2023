# Deep semi-supervised Learning using Autoencoder

The original code is from a github repository at: https://github.com/vivekamin/semi-supervised-learning

## Data
Considered 1000 labeled datapoints from MNIST with 100 labeled samples from each digit. Used the rest of the training images as unlabeled data. Trained the sparse autoencoder with p=0.1 and extracted the hidden representations for these labeled samples to get a matrix with dimensions [D × m] = [200 × 1000]. 

## Models
Classifier 1:
Trained a softmax classifier with these feature representations to classify the digits. Combined the encoder and the softmax classifier to create a classifier with network dimensions [784, 200, 10]. 

Classifier 2:
Created another fully connected network with same dimensions [784, 200, 10] and initialize it with random values. Trained it with 1000 labeled samples from MNIST. 

Using these two networks, computed the classification accuracies for the test data. To compare the performance between them, in this competition the classifier 1 won.

## how to run the code
1. run 'download_mnist.sh'
2. run 'sparseae_ssl.py'
