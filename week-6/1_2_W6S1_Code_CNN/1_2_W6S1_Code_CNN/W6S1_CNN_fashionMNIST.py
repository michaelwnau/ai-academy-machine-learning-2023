#!/usr/bin/env python
# coding: utf-8

# ## Week 6 - Session 1: Convolutional Neural Networks (CNN) with fashion-MNIST

# In[1]:


# 0. Import TensorFlow
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


# ### Question 1 >>
#  1. Download the Fashion-MNIST dataset
#      - For more information >> https://github.com/zalandoresearch/fashion-mnist
#  2. Normalize pixel values to be between 0 and 1

# In[2]:


# Download data  
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# >> Print out the shape of dataset
print("train_images.shape = {}".format(train_images.shape))
print("train_labels.shape = {}".format(train_labels.shape))
print("test_images.shape = {}".format(test_images.shape))
print("test_labels.shape = {}\n".format(test_labels.shape))


# * 60,000 28x28 images for training data and 10,000 images for test data

# 3. Verify the data

# In[3]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# 4. Add a channels dimension

# In[4]:


train_images = train_images[..., tf.newaxis].astype("float32")
test_images = test_images[..., tf.newaxis].astype("float32")

# >> Print out the shape of dataset
print("train_images.shape = {}".format(train_images.shape))
print("train_labels.shape = {}".format(train_labels.shape))
print("test_images.shape = {}".format(test_images.shape))
print("test_labels.shape = {}\n".format(test_labels.shape))


# ### Question 2 >>
# - Set up a CNN-based image classification model
# - What is CNN?  Explain in a couple sentences.

# In[5]:


model = models.Sequential()
model.add(layers.Conv2D(32, 2, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, 2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model.summary()


# In[7]:


get_ipython().run_cell_magic('time', '', "# 6. Compile and train the model\nmodel.compile(optimizer='adam',\n              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n              metrics=['accuracy'])\n\nhistory = model.fit(train_images, train_labels, epochs=10, batch_size=64, \n                    validation_data=(test_images, test_labels))")


# In[8]:


# 7. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

# >> Print out test acc/loss
print("Test Accuracy > {}".format(test_acc))
print("Test Loss > {}".format(test_loss))


# In[9]:


# 7. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

# >> Print out test acc/loss
print("Test Accuracy > {}".format(test_acc))
print("Test Loss > {}".format(test_loss))


# ### Question 3 >>

# In[10]:


# >> Plot loss
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# >> Plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.tight_layout()

