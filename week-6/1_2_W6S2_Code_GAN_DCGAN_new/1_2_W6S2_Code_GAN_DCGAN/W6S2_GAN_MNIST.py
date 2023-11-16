#!/usr/bin/env python
# coding: utf-8

# ## Week 6 - Session 2: Generative adversarial network (GAN)
# * Install package: imageio
# 
# * Development environment: 
#         cuda 8.0
#         Python 3.5.3
#         tensorflow-gpu 1.2.1
#         numpy 1.13.1
#         matplotlib 2.0.2
#         imageio 2.2.0
# 
# * if you have a later version of tensorflow, there could be deprecation warnings.
# 
# """
# Source: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN
# """

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   

import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data

tf.compat.v1.disable_eager_execution()


# ### Generator & Discriminator
# 1. What is the role of generator? How does it work? What is the output?
# 2. What is the role of discriminator? How does it work? What is the output?
# 3. What can we do with GAN?

# In[2]:


# G(z)
def generator(x):
    # initializers
    w_init = tf.compat.v1.initializers.random_normal(mean=0, stddev=0.02)
    b_init = tf.compat.v1.initializers.constant(0.0)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.compat.v1.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('G_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
    b2 = tf.compat.v1.get_variable('G_b2', [1024], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.compat.v1.get_variable('G_w3', [h2.get_shape()[1], 784], initializer=w_init)
    b3 = tf.compat.v1.get_variable('G_b3', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

    return o


# D(x)
def discriminator(x, drop_out):

    # initializers
    w_init = tf.compat.v1.initializers.random_normal(mean=0, stddev=0.02)
    b_init = tf.compat.v1.initializers.constant(0.0)

    # 1st hidden layer
    w0 = tf.compat.v1.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.compat.v1.get_variable('D_b0', [1024], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.compat.v1.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.compat.v1.get_variable('D_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    w2 = tf.compat.v1.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.compat.v1.get_variable('D_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.compat.v1.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
    b3 = tf.compat.v1.get_variable('D_b3', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3)

    return o


# ### Functions for showing Training/ Results

# In[3]:


fixed_z_ = np.random.normal(0, 1, (25, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = np.random.normal(0, 1, (25, 100))

    if isFix:
        test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
    else:
        test_images = sess.run(G_z, {z: z_, drop_out: 0.0})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# ### Load Data

# In[4]:


# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # old version
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1


# ### Define GAN: networks, loss function, and optimizer

# In[5]:


lr = 0.0002

# networks : generator
with tf.compat.v1.variable_scope('G'):
    z = tf.compat.v1.placeholder(tf.float32, shape=(None, 100))
    G_z = generator(z)

# networks : discriminator
with tf.compat.v1.variable_scope('D') as scope:
    drop_out = tf.compat.v1.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 784))
    D_real = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake = discriminator(G_z, drop_out)


# loss for each network
eps = 1e-2
# D: maximize D_real and (1-D_fake) (=minimize - log (D_real) - log (1-D_fake)
D_loss = tf.reduce_mean(-tf.math.log(D_real + eps) - tf.math.log(1 - D_fake + eps)) 
# G: maximize the expectation of D_fake
G_loss = tf.reduce_mean(-tf.math.log(D_fake + eps))  

# trainable variables for each network
t_vars = tf.compat.v1.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.compat.v1.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.compat.v1.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)


# ### GPU setting
#  - NVIDIA GPU example

# In[ ]:


# open session and initialize all variables

if True: # If you have multiple GPUs, setup GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'          # GPU-ID "0" or "0, 1" for multiple
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 
    # config.gpu_options.allow_growth = True 
    
    sess = tf.compat.v1.InteractiveSession(config=config)
else:
    sess = tf.compat.v1.InteractiveSession()


# ### GAN Training
# - Review the given output for discussion. The given output was generated with 100 epochs.
# - Please do not wait to finish training GAN during the workshop. It took ~12 min to train 100 epochs with a GPU (Titan XP). 
# - When training it, you may reduce the training epoch. 

# In[10]:


# training parameters
batch_size = 100
train_epoch = 40  # !!!!! The below results were generated with 100 epochs

tf.compat.v1.global_variables_initializer().run()

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
    os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
    os.mkdir('MNIST_GAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(train_set.shape[0] // batch_size):
        
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        # x_ = x_.reshape([x_.shape[0],-1])

        z_ = np.random.normal(0, 1, (batch_size, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch + 1), show=True, save=False, path=p, isFix=False)
    show_result((epoch + 1), show=True, save=False, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

sess.close()


# ### Questions
# 1. Check the results in : ./MNIST_GAN_results/ (Fixed_results/ and generation_animation.gif)
# 2. Compare the loss of discriminator (loss_d) with the loss of generator (loss_g) during training. What kinds of trends do you observe? What does it mean? 
# 
#         loss graph: ./MNIST_GAN_results/MNIST_GAN_train_hist.png 
# 
# 3. Can you distinguish the (fake) digit images generated by GAN from the real ones? 
