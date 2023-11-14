#!/usr/bin/env python
# coding: utf-8

# ## Week 6 - Session 1: Convolutional Neural Networks (CNN)
# * Download the CIFAR10 dataset
# * Normalize pixel values to be between 0 and 1

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
numpy.random.seed(2)
tf.random.set_seed(2)

import ssl #this line seems to be necessary for windows users
ssl._create_default_https_context = ssl._create_unverified_context


# ### After running the above cell, answer the following questions:
#   
#   1. What is the CIFAR datset?
#   2. What is the purpose for creating the dataset?

# In[2]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Loding CIFAR10 data... Done!")
print("train_images.shape = {}".format(train_images.shape))
print("train_labels.shape = {}".format(train_labels.shape))
print("test_images.shape = {}".format(test_images.shape))
print("test_labels.shape = {}\n".format(test_labels.shape))

train_images, test_images = train_images / 255.0, test_images / 255.0


# * Compared to Fashion-MNIST dataset, CIFAR-10 dataset has the following differences.
#   1. Different training data size: 50,000 (CIFAR-10) vs 60,000 (Fashion-MNIST)
#   2. Different image size: 32x32 (CIFAR-10) vs 28x28 (Fashion-MNIST)
#   3. Different channel: 3 RGB channel (CIFAR-10) vs 1 grayscale channel (Fashion-MNIST)

#  * Verify the data

# In[3]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# * Set up two CNN-based image classification models

# In[4]:


# model 1: Baseline model 
model1 = models.Sequential()
model1.add(layers.Conv2D(32, 2, activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, 2, activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(10))

# model 2: Deeper CNN with dropout
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model2.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Dropout(0.2))
model2.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Dropout(0.2))
model2.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Dropout(0.2))
model2.add(layers.Flatten())
model2.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(10))


# ### After the above code runs, answer the following questions:
# - How are the above models the same/different?

# * Compile and train the baseline model

# In[5]:


get_ipython().run_cell_magic('time', '', "# Model 1\nmodel1.compile(optimizer='adam',\n              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n              metrics=['accuracy'])\n\nhistory1 = model1.fit(train_images, train_labels, epochs=10, batch_size=64,\n                    validation_data=(test_images, test_labels))")


# ### As the above code and below code executes, answer the following questions:
# 
# 1. How is the accuracy changing as each epoch executes?
# 2. How is the loss changing?
# 3. what is meant by loss, val_loss, and val_accuracy?

# In[6]:


get_ipython().run_cell_magic('time', '', "# Model 2 \nmodel2.compile(optimizer='adam',\n              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n              metrics=['accuracy'])\n\nhistory2 = model2.fit(train_images, train_labels, epochs=10, batch_size=64,\n                    validation_data=(test_images, test_labels))")


# * Plot the training curves for both models.
# * Evaluate the results and compare the performance of two models.

# In[7]:


get_ipython().run_cell_magic('time', '', '# Model 1\nmodel = model1\nhistory = history1\n\ntest_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)\n\n# print out test acc/loss\nprint("Evaluation of the baseline model ---\\n")\nprint("Test Accuracy > {}".format(test_acc))\nprint("Test Loss > {}".format(test_loss))\n\n# plot loss\nplt.subplot(211)\nplt.title(\'Cross Entropy Loss\')\nplt.plot(history.history[\'loss\'], color=\'blue\', label=\'train\')\nplt.plot(history.history[\'val_loss\'], color=\'orange\', label=\'test\')\nplt.xlabel(\'Epoch\')\nplt.ylabel(\'Loss\')\nplt.legend(loc=\'upper right\')\n\n# plot accuracy\nplt.subplot(212)\nplt.title(\'Classification Accuracy\')\nplt.plot(history.history[\'accuracy\'], color=\'blue\', label=\'train\')\nplt.plot(history.history[\'val_accuracy\'], color=\'orange\', label=\'test\')\nplt.xlabel(\'Epoch\')\nplt.ylabel(\'Accuracy\')\nplt.legend(loc=\'lower right\')\nplt.tight_layout()')


# ### After running the above code, answer the following questions.
# 
# 
# 1. What are these graphs telling us?
# 2. At what epoch do we see the best results?
# 

# In[8]:


get_ipython().run_cell_magic('time', '', '# Model 2\nmodel = model2\nhistory = history2\n\ntest_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)\n\n# print out test acc/loss\nprint("Evaluation of the NEW model ---\\n")\nprint("Test Accuracy > {}".format(test_acc))\nprint("Test Loss > {}".format(test_loss))\n\n# plot loss\nplt.subplot(211)\nplt.title(\'Cross Entropy Loss\')\nplt.plot(history.history[\'loss\'], color=\'blue\', label=\'train\')\nplt.plot(history.history[\'val_loss\'], color=\'orange\', label=\'test\')\nplt.xlabel(\'Epoch\')\nplt.ylabel(\'Loss\')\nplt.legend(loc=\'upper right\')\n\n# plot accuracy\nplt.subplot(212)\nplt.title(\'Classification Accuracy\')\nplt.plot(history.history[\'accuracy\'], color=\'blue\', label=\'train\')\nplt.plot(history.history[\'val_accuracy\'], color=\'orange\', label=\'test\')\nplt.xlabel(\'Epoch\')\nplt.ylabel(\'Accuracy\')\nplt.legend(loc=\'lower right\')\nplt.tight_layout()')


# ### Additional Questions
# 1. A deeper neural networks is always better?
# 2. What is the role of dropout in neural networks?
