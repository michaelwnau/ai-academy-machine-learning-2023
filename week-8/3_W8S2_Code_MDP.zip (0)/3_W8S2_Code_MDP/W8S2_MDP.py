#!/usr/bin/env python
# coding: utf-8

# ## Week8 - Session 2: Markov Decision Process (MDP) - Random Features

# install mdptoolbox: pip install pymdptoolbox or refer to the link: https://pymdptoolbox.readthedocs.io/en/latest/

# In[1]:


import numpy as np
import pandas as pd
import mdptoolbox, mdptoolbox.example
import argparse
import math
from MDP_policy import *
import random
import warnings
warnings.filterwarnings('ignore')


# ### Part 1. Load Data

# In[2]:


# Load data
org_data = pd.read_csv('MDP_data_student200.csv')

# Define different feature sets: feature_space(total), static_feats, candidate_feats
feature_space = list(org_data.columns.values)
static_feats, candidate_feats = feature_space[:6], feature_space[6:]

# Report the number of features.
print("# static features: {}".format(len(static_feats)))
print("# candidate features: {}\n".format(len(candidate_feats)))

org_data.head(3)


# ### Part 2. Discretization

# In[3]:


# Extract continuous features for discretization
#  - we consider features with more than 6 distinct values continous, except for the 'object' type features.

# > get column names, types, # of unique values
col_names = list(org_data.columns.values)
col_types = list(org_data.dtypes.astype('str'))

col_numval = []
for col in col_names:
    numval = len(org_data[col].unique())
    col_numval.append(numval)

# > generate a dataframe for extraction
col_dict = {'name': col_names, 'type': col_types, 'uniqueval': col_numval}
col_data = pd.DataFrame(col_dict, columns=['name', 'type', 'uniqueval'])

# > extract features which satisfy the condition
n_bins = 2
condition = (col_data['uniqueval'] > n_bins)
continuous_feats = col_data[condition].name.tolist()
continuous_feats = [i for i in continuous_feats if i not in static_feats]

# > print the number of selected features
print("# continuous features: {}".format(len(continuous_feats)))

# Discretize the continuous features 
from sklearn.preprocessing import KBinsDiscretizer
enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')#, strategy='uniform')

new_data = org_data.copy()

# for col in continuous_feats:
for col in candidate_feats:
    data_reshaped = np.reshape(np.array(org_data[col]), (-1, 1))
    new_data[col] = enc.fit_transform(data_reshaped)
    
new_data.head(3)


# ### Part 3. Feature Selection

# In[4]:


get_ipython().run_cell_magic('time', '', 'x = []\ny = []\nfeature_sets = {}\ncan_size = len(candidate_feats)\n\n# line 1-3\nfor k in range(1, 9):\n    print("{} selected:".format(k), end=" ")\n    # random select k features from the candidate feature set.\n    chosen_feats = random.sample(candidate_feats, k)\n    # store the chosen feature set in a dictionary\n    feature_sets[k] = chosen_feats\n    print(chosen_feats, end=" ")\n    \n    sample_feats = static_feats + chosen_feats\n    sample_data = new_data[sample_feats]\n    \n    # load data set with selected or extracted discrete features\n    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)\n    \n    # apply Value Iteration to run the MDP\n    try:\n        vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)\n        vi.run()\n\n        # evaluate policy using ECR\n        ecr = calcuate_ECR(start_states, vi.V)\n        print("-> ECR: {}\\n".format(ecr))\n        \n        x.append(k)\n        y.append(ecr)\n        \n    except OverflowError:\n        print("Error occured!\\n")\n        pass')


# ### Result : Random Feature Selection

# In[5]:


import matplotlib.pyplot as plt

plt.plot(x, y, 'ro-')
plt.title('[Result] Random Feature Selection')
plt.xlabel('Number of features')
plt.ylabel('ECR')
plt.show()


# ### Part 4. Induce Policy with the best feature set

# In[6]:


# feature counts with the highest ECR.
best_k = 8

sample_feats = static_feats + feature_sets[best_k]
sample_data = new_data[sample_feats]

# load data set with selected or extracted discrete features
[start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)

# apply Value Iteration to run the MDP
try:
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)
    vi.run()
    
    # output policy
    output_policy(distinct_acts, distinct_states, vi)

except OverflowError:
    print("Error occured!\n")
    pass

