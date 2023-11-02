#!/usr/bin/env python
# coding: utf-8

# ## Week 5 - Session 2: SSL_AutoEncoder
# - Download the data under the working directory, using shell script: $ sh download_mnist.sh

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from load_dataset import mnist

import matplotlib.pyplot
import pdb
import tensorflow as tf
import math

tf.compat.v1.disable_eager_execution()


# ### Fully-connected network (feed-forward network)

# In[2]:


def fc_networks(epochs, learning_rate):
    
    # input layer: size of digit image (784 = 28 x 28)
    x_fc = x = tf.compat.v1.placeholder(tf.float32, [None, 784]) 

    # hidden layer: 200 hidden units
    W1_fc = tf.compat.v1.Variable(tf.random.normal([784, 200], stddev=0.03), name='W1_FC')
    b1_fc = tf.compat.v1.Variable(tf.random.normal([200]), name='b1_fc')
    l1_linear_output = tf.add(tf.matmul(x_fc, W1_fc), b1_fc) 
    l1_activation = tf.compat.v1.nn.relu(l1_linear_output) 
    
    # output layer: multi-classes (10 digits)
    W2_fc = tf.compat.v1.Variable(tf.random.normal([200, 10], stddev=0.03), name='W2_FC')
    b2_fc = tf.compat.v1.Variable(tf.random.normal([10]), name='b2_fc')  
    # softmax: probability distribution of output classes
    y_fc = tf.compat.v1.nn.softmax(tf.add(tf.matmul(l1_activation, W2_fc), b2_fc))  

    # cost function
    y_label_input_fc = tf.compat.v1.placeholder(tf.float32, [None, 10]) # true labels
    cross_entropy_fc = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_label_input_fc, logits=y_fc))
    
    # Training with an optimizer
    fc_classifier = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                        beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cross_entropy_fc)
    
    return x_fc, y_label_input_fc, y_fc, cross_entropy_fc, fc_classifier


# ### Looking at the above function, answer the following questions:
# 
# 1. What is a feed-forward network.  Explain in a couple of sentences.
# 2. What are some advantages to a feed-forward network?

# ### (Unsupervised) Sparse autoencoder  + (Supervised) NN Classifier = (Semi-Supervised) NN classifier
# * Autoencoder is an unsupervised learning technique for neural networks to automatically learn features from unlabeled data.
# * Sparse constraints can significantly save computing resources and find the characteristics of data in a low-dimensional space
# 
# --> Sparse autoencoder =  efficiently train latent features from unlabeled data in a low dimensional space
# 
# 

# ### Sparse autoecoder with classification: High-level Algorithm

#  1. (unsupervised) Train latent representation (extracted features) from all the data (labeled + unlabeled)
#    - Encoding: the extracted (and abstracted) features from the whole data (L+U) are more general than the features only from labeled data (L).
#    - Decoding: make sure the decoded output from latent representation close to the input
#  2. (supervised) Train a classifier using the latent representation and labels from labeled data

# ### AutoEncoder Network Structure

# input (x) -- Encoder (hidden layer) -- [latent features] -- Decorder (x_hat)
#  
#                                       \ Classifier (y_label)
# 

# In[3]:


def sparse_autoEncoder(learning_rate, reg_term_lambda, p, beta):
    
    # 1. train an unsupervised auto-encoder on the whole training data (L + U)
    # x: input, x_hat: decoded output
    # encoded variables: latent representation 
    x = tf.compat.v1.placeholder(tf.float32, [None, 784]) # 784 = 28 * 28 (size of a digit image)
    x_hat = tf.compat.v1.placeholder(tf.float32, [None, 784])
    y_label_input = tf.compat.v1.placeholder(tf.float32,[None,10])

    # Encoder: First hidden layer
    W1 = tf.compat.v1.Variable(tf.random.normal([784, 200], stddev=0.03), name='W1')  # num of hidden unit: 200
    b1 = tf.compat.v1.Variable(tf.random.normal([200]), name='b1')
    # Output from hidden layer 1
    linear_layer_one_output = tf.add(tf.matmul(x, W1), b1)
    layer_one_output = tf.compat.v1.nn.sigmoid(linear_layer_one_output) # Activation function: sigmoid

    # Decorder
    W2 = tf.compat.v1.Variable(tf.random.normal([200, 784], stddev=0.03), name='W2')  # output size = input size
    b2 = tf.compat.v1.Variable(tf.random.normal([784]), name='b2')
    # x_hat : reconstructed output from latent representation 
    linear_layer_two_output = tf.add(tf.matmul(layer_one_output, W2), b2)
    x_hat = tf.compat.v1.nn.sigmoid(linear_layer_two_output)

    # Softmax classifier weight initialization
    W3 = tf.compat.v1.Variable(tf.random.normal([200, 10], stddev=0.03), name='W3')   # 
    b3 = tf.compat.v1.Variable(tf.random.normal([10]), name='b3')

    # 2. Connect softmax with feature extractor:
    # Use the latent representations extracted from this auto-encoder 
    # to train a softmax classier on labeled proportion of data.
    y_label = tf.compat.v1.nn.softmax(tf.add(tf.matmul(layer_one_output, W3), b3))

    
    # Define classifier
    cross_entropy = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_label_input, logits=y_label))
    
    softmax_classifier = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, 
                                                          beta2=0.999, epsilon=1e-08).minimize(cross_entropy)
    
    # p_hat : latent representation  
    p_hat = tf.reduce_mean(tf.compat.v1.clip_by_value(layer_one_output, 1e-10, 1.0), axis=0)

    # p_hat = tf.reduce_mean(layer_one_output,axis=1)
    kl = kl_divergence(p, p_hat)

    # Define the cost function for the decoder
    # 1) diff : difference between reconstruected output x_hat and input x
    # 2) reg_term_lambda (weight decay parameter): decrease the magnitude of the weights, 
    #                                           and helps prevent overfitting
    # 3) beta: controls the weight of the sparsity penalty term
    diff = x_hat - x   
    cost = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) + reg_term_lambda * (
                tf.compat.v1.nn.l2_loss(W1) + tf.compat.v1.nn.l2_loss(W2)) + beta * tf.reduce_sum(kl)

    optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, 
                                                 epsilon=1e-08).minimize(cost)

    init_op = tf.compat.v1.global_variables_initializer()
    
    return x, y_label_input, y_label, cost, optimiser, init_op, cross_entropy, softmax_classifier
    
    
# KL divergence: a standard function for measuring how different two different distributions are.
# if KL == 0, p = p_hat
# otherwise, it increases monotonically as p_hat diverges from p.
def kl_divergence(p, p_hat):
    return p * tf.math.log(p) - p * tf.math.log(p_hat) + (1 - p) * tf.math.log(1 - p) - (1 - p) * tf.math.log(1 - p_hat)


# ### Looking at the code above, answer the following questions:
# 
# 1. What is the autoencoder doing?
# 2. Why is it necessary to use this?

# ### Data preperation
#  - get a subset of 100 data for each digit as labeled training data

# In[4]:


def get_Subset_LabeledData(train_data, train_label, test_data, test_label):
    digits = [0,0,0,0,0,0,0,0,0,0]
    index_to_get = []
    # Get the subset dataset: 100 data for each digit
    index_l = 0
    for i in range(0,train_label.shape[1]):
        #print (train_label[0][i])
        if digits[int(train_label[0][i])] == 100:
            continue
        else:
            digits[int(train_label[0][i])]+=1
            index_to_get.append(index_l)

        index_l += 1

    # Labeled samples    
    train_data_new = train_data.take(index_to_get,axis=1)
    train_label_new = train_label.take(index_to_get)

    test_label_new = test_label[0]

    n_values = np.max(train_label_new.astype(int)) + 1              # number of digits
    train_label_new = np.eye(n_values)[train_label_new.astype(int)] # one-hot encoded labels

    n_values = np.max(test_label_new.astype(int)) + 1               # number of digits
    test_label_new = np.eye(n_values)[test_label_new.astype(int)]   # one-hot encoded labels
    
    return train_data_new, train_label_new, test_data, test_label_new


# In[5]:


def train_fc(epochs, ntrain):
    
    # Load the data
    train_data, train_label, test_data, test_label = mnist(ntrain, ntest=1000, digit_range=[0, 10])
    
    # Select 100 labeled sample for each digit from the training data (L), 
    # and use the rest of the training set as the unlabeled data (U).    
    train_data_new, train_label_new, test_data_new, test_label_new = get_Subset_LabeledData(train_data, train_label, 
                                                                                      test_data, test_label)

    print ("labeled data: train {}, test {}".format(train_label_new.shape, test_label_new.shape))

    # Define a fully-connected networks
    x_fc, y_label_input_fc, y_fc, cross_entropy_fc, fc_classifier = fc_networks(epochs, learning_rate=1e-3)   

    # Define a sparse autoencoder
    x, y_label_input, y_label, cost, optimiser, init_op, cross_entropy, softmax_classifier = sparse_autoEncoder(
        learning_rate=1e-3, reg_term_lambda=1e-3, p=0.1, beta=3)
    

    with tf.compat.v1.Session() as sess:
        # initialise the variables
        sess.run(init_op)

        # FC
        print("Training FC network using the labeled data (L)")
        for epoch in range(epochs):
            cost_fc = 0
            _ , cost_fc = sess.run([fc_classifier, cross_entropy_fc],
                                   feed_dict={x_fc: train_data_new.T, 
                                              y_label_input_fc: train_label_new})
            if ((epoch + 1) % 200 == 0):
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost_fc))
                
        correct_prediction_fc = tf.equal(tf.argmax(y_label_input_fc, 1), tf.argmax(y_fc, 1))
        accuracy_fc = tf.reduce_mean(tf.cast(correct_prediction_fc, tf.float32))                
        print("\nAccuracy of FC network is", sess.run(accuracy_fc, 
                                        feed_dict={x_fc: test_data_new.T, 
                                                   y_label_input_fc: test_label_new}))


# ### Train AutoEncoder

# In[6]:


def train_autoencoder(epochs, ntrain):
    
    # Load the data
    train_data, train_label, test_data, test_label = mnist(ntrain, ntest=1000, digit_range=[0, 10])
    
    # Select 100 labeled sample for each digit from the training data (L), 
    # and use the rest of the training set as the unlabeled data (U).    
    train_data_new, train_label_new, test_data_new, test_label_new = get_Subset_LabeledData(train_data, train_label, 
                                                                                      test_data, test_label)

    print ("labeled data: train {}, test {}".format(train_label_new.shape, test_label_new.shape))

    # Define a fully-connected networks
    x_fc, y_label_input_fc, y_fc, cross_entropy_fc, fc_classifier = fc_networks(epochs, learning_rate=1e-3)   

    # Define a sparse autoencoder
    x, y_label_input, y_label, cost, optimiser, init_op, cross_entropy, softmax_classifier = sparse_autoEncoder(
        learning_rate=1e-3, reg_term_lambda=1e-3, p=0.1, beta=3)
    

    with tf.compat.v1.Session() as sess:
        # initialise the variables
        sess.run(init_op)             

        print("\nTraining Sparse Autoencoder using the whole data (L + U)")
        for epoch in range(epochs):
            _, c = sess.run([optimiser, cost], feed_dict={x: train_data.T})            
            if((epoch+1)%200 == 0):
                print("Epoch:", (epoch + 1), "cost = {:.3f}".format(c))

        print("Training a softmax classifier with Autoencoder latent representations, using the labeled data (L)")
        for epoch in range(epochs):
            cost = 0
            _ , cost = sess.run([softmax_classifier, cross_entropy], feed_dict={x: train_data_new.T, y_label_input : train_label_new})
            if ((epoch + 1) % 200 == 0):
                print("Epoch:", (epoch + 1), "cost = {:.3f}".format(cost))


        correct_prediction = tf.equal(tf.argmax(y_label_input,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print("Accuracy of Sparse Autoencoder is", sess.run(accuracy, 
                                      feed_dict={x: test_data_new.T, y_label_input : test_label_new}))


# ### Questions
# 1. What is the cost telling us?
# 2. What is the epoch telling us?
# 3. Compare the prediction performance between Supervised Fully-connected networks and Semi-supervised AutoEncoder.
# 4. Change the number of unlabeled data, and observe the results
#    - The labeled data is fixed with 1000 instances.
#    - To change the number of unlabeld data, change 'ntrain'. (i.e., unlabeled data num = ntrain - 1000) 

# In[9]:


if __name__ == "__main__":
    epochs = 1000
    ntrain = 6000
    train_fc(epochs, ntrain)
    train_autoencoder(epochs, ntrain)


# In[ ]:




