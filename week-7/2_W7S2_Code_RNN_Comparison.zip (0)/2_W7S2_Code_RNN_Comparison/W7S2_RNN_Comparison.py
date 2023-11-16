#!/usr/bin/env python
# coding: utf-8

# ## Week 7 - Session 1: Compare RNNs

# ### Initialization

# In[15]:


# Initialize modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
np.random.seed(13)
tf.random.set_seed(13)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Declare necessary functions
def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def baseline(history):
  return np.mean(history)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure(figsize=(6,4))

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


# ### Part 1: Prepare the dataset

# In[2]:


# Load dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
df.head()


# In[3]:


# Extract features
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]
features.index = df['Date Time']

# Plot features
features.plot(subplots=True)


# * Atmospheric pressure, p (mbar), doesn't show any patterns from the plot and its value varies a little compared to other two features.
# * Air temperature and air density show specific pattern (seasonality) and their values vary more.

# In[4]:


# open session and initialize all variables

if True: # If you have multiple GPUs, you can set up how to use GPUs. Otherwise, set False
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'           # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'         # GPU-ID "0" or "0, 1" for multiple
    config = tf.compat.v1.ConfigProto()   
    config.gpu_options.per_process_gpu_memory_fraction = 0.2 
    #config.gpu_options.allow_growth = True 
    
    sess = tf.compat.v1.InteractiveSession(config=config)
else:  # If you do not have a GPU
    sess = tf.compat.v1.InteractiveSession()


# In[5]:


# Initialize params
TRAIN_SPLIT = 300000
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 200
EPOCHS = 10
past_history = 720
future_target = 72
STEP = 6

# Standardize the dataset
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# Prepare the dataset for the prediction task
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# ### Part 2: Set-up and Train RNN-based Models

# ### RNN

# In[6]:


# Set up the baseline model
rnn = tf.keras.models.Sequential()
rnn.add(tf.keras.layers.SimpleRNN(32, input_shape=x_train_single.shape[-2:]))
rnn.add(tf.keras.layers.Dense(1))

rnn.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

rnn.summary()


# In[7]:


get_ipython().run_cell_magic('time', '', '# Train RNN\nrnn_history = rnn.fit(train_data_single, epochs=EPOCHS,\n                                            steps_per_epoch=EVALUATION_INTERVAL,\n                                            validation_data=val_data_single,\n                                            validation_steps=50)')


# ### LSTM

# In[8]:


# Set up the LSTM model
lstm = tf.keras.models.Sequential()
lstm.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
lstm.add(tf.keras.layers.Dense(1))

lstm.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

lstm.summary()


# In[9]:


get_ipython().run_cell_magic('time', '', '# Train LSTM\nlstm_history = lstm.fit(train_data_single, epochs=EPOCHS,\n                                            steps_per_epoch=EVALUATION_INTERVAL,\n                                            validation_data=val_data_single,\n                                            validation_steps=50)')


# ### GRU

# In[10]:


# Set up the GRU model
gru_model = tf.keras.models.Sequential()
gru_model.add(tf.keras.layers.GRU(32, input_shape=x_train_single.shape[-2:]))
gru_model.add(tf.keras.layers.Dense(1))

gru_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
gru_model.summary()


# In[11]:


get_ipython().run_cell_magic('time', '', '# Train GRU\ngru_history = gru_model.fit(train_data_single, epochs=EPOCHS,\n                            steps_per_epoch=EVALUATION_INTERVAL,\n                            validation_data=val_data_single,\n                            validation_steps=50)')


# In[16]:


# Plot losses for LSTM/GRU model
plot_train_history(rnn_history, 'Loss - RNN')
plot_train_history(lstm_history, 'Loss - LSTM')
plot_train_history(gru_history, 'Loss - GRU')


# In[18]:


# Compare the MAEs: LSTM vs GRU
rnn_loss = rnn.evaluate(val_data_single, steps=50, verbose=0) 
lstm_loss = lstm.evaluate(val_data_single, steps=50, verbose=0) 
gru_loss = gru_model.evaluate(val_data_single, steps=50, verbose=0) 

print("RNN: {:.4f} vs. LSTM: {:.4f} vs. GRU: {:.4f}".format(rnn_loss, lstm_loss, gru_loss))


# ### Part 3: Tuning the Model

# In[20]:


get_ipython().run_cell_magic('time', '', 'tune_histories = []\ntune_losses = []\n\n# Initialize params\nTRAIN_SPLIT = 300000\nBATCH_SIZE = 256\nBUFFER_SIZE = 10000\nEVALUATION_INTERVAL = 200\nEPOCHS = 10\npast_histories = [720, 500, 150]  # length of past history \nfuture_target = 72\nSTEP = 6\n\nfor past_history in past_histories:\n  # Prepare the dataset for the prediction task\n    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,\n                                                    TRAIN_SPLIT, past_history,\n                                                    future_target, STEP,\n                                                    single_step=True)\n    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],\n                                                TRAIN_SPLIT, None, past_history,\n                                                future_target, STEP,\n                                                single_step=True)\n\n    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))\n    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()\n\n    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))\n    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()\n\n    # Set up the GRU model\n    gru_tune = tf.keras.models.Sequential()\n    gru_tune.add(tf.keras.layers.GRU(32, input_shape=x_train_single.shape[-2:]))\n    gru_tune.add(tf.keras.layers.Dense(1))\n\n    gru_tune.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=\'mae\')\n\n    # Train the model\n    print("Training a model with {} past history >>".format(past_history))\n    gru_tune_history = gru_tune.fit(train_data_single, epochs=EPOCHS,\n                              steps_per_epoch=EVALUATION_INTERVAL,\n                              validation_data=val_data_single,\n                              validation_steps=50, verbose=0)\n  \n    tune_histories.append(gru_tune_history)\n    tune_losses.append(gru_tune.evaluate(val_data_single, steps=50, verbose=0))\n\nprint("Done!")')


# In[21]:


for idx, hist in enumerate(tune_histories):
  plot_train_history(hist, 'Past history = {}'.format(past_histories[idx]))
  print(">> Test loss(MAE) = {}\n".format(tune_losses[idx]))


# ## After running the above code, think about or implement the following:
# 1. Compare and contrast the different models presented above.
# 2. Try passing some other "features considered" into the model and see how it performs.
