#!/usr/bin/env python
# coding: utf-8

# ## Week 9 - Session 1: MDP with PCA
# 
# - Understand three types of feature selection methods: Filtered, Wrapper, and Embedded Approach
# - Understand PCA as a feature selection method (advantages and limitations) 
# - Understand how to compare two different feature selection approaches (PCA vs. Random Feature Selection) with an MDP framework in terms of effectiveness and efficiency

# In[1]:


import numpy as np
import pandas as pd
import mdptoolbox, mdptoolbox.example
import argparse
import math
from MDP_policy import *
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import random 
import warnings
warnings.filterwarnings('ignore')


# ## MDP with PCA feature selection
# ### 1.1. Load Data and Preprocessing

# In[2]:


# Load data
org_data = pd.read_csv('./MDP_dataset.csv')

# define different feature sets: feature_space(total), static_feats, candidate_feats
feature_space = list(org_data.columns.values)
static_feats, candidate_feats = feature_space[:6], feature_space[6:]

print("# static features: {}".format(len(static_feats)))
print("# candidate features: {}\n".format(len(candidate_feats)))

org_data.head(3)


# ### 2.2. Feature Selection with PCA + MDP training (Value Iteration)
#  - Set the same number of bins to the random feature selecction for the comparison
#  - Explore different number of bins and compare the results of two approaches (random feature selection and PCA) to examine the robustness of each approach.  

# In[3]:


get_ipython().run_cell_magic('time', '', 'x = []\ny = []\ndata_sets = {}\ncandidate_data = org_data[candidate_feats]\n\nn_bins = 7\nfor k in range(1, 9):\n    steps = [\n             (\'standardize\', StandardScaler()),\n             (\'pca\', PCA(k)),\n             (\'discretize\', KBinsDiscretizer(n_bins=n_bins, encode=\'ordinal\'))\n            ]\n    \n    model = Pipeline(steps=steps)\n\n    data = model.fit_transform(candidate_data)\n    \n    col_names = [\'f{}\'.format(n) for n in range(1, k+1)]\n    new_data = pd.DataFrame(data=data, columns=col_names)\n    \n    sample_data = pd.concat([org_data[static_feats], new_data], axis=1)\n    sample_feats = list(sample_data.columns.values)\n    data_sets[k] = sample_data\n    \n    # load data set with selected or extracted discrete features\n    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)\n    \n    # apply Value Iteration to run the MDP\n    try:\n        vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)\n        vi.run()\n\n        # ----------------------------------------------\n        # evaluate policy using ECR\n        ecr = calcuate_ECR(start_states, vi.V)\n        print("{} Features - ECR {}".format(k, ecr))\n        \n        x.append(k)\n        y.append(ecr)\n        \n    except OverflowError:\n        error_feats.append(feat)\n        print("Error occured!")\n        pass')


# ### 2.4. Results

# In[4]:


import matplotlib.pyplot as plt

plt.plot(x, y, 'go-')
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.title('[Result] PCA')
plt.xlabel('Number of features')
plt.ylabel('ECR')
plt.show()


# ### 2.5. Value Iteration with the best feature set (PCA)

# In[7]:


get_ipython().run_cell_magic('time', '', '# feature counts with the highest ECR.\nbest_k = y.index(max(y))+1\nprint("Best k (number of features): {}".format(best_k))\n\nsample_data = data_sets[best_k]\nsample_feats = list(sample_data.columns.values)\n\n# load data set with selected or extracted discrete features\n[start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)\n\n# apply Value Iteration to run the MDP\ntry:\n    vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)\n    vi.run()\n    \n    # output policy\n    output_policy(distinct_acts, distinct_states, vi)\n\nexcept OverflowError:\n    print("Error occured!\\n")\n    pass\n\nn_states = n_bins\n\nfor i in range(1, best_k):\n    n_states *= n_bins\n    \nprint("\\nTotal number of states: {}".format(n_states))')


# ## MDP with Random Feature Selection 

# ### 1. Load data and preprocession

# In[8]:


# Load data
org_data = pd.read_csv('MDP_dataset.csv')

# Define different feature sets: feature_space(total), static_feats, candidate_feats
feature_space = list(org_data.columns.values)
static_feats, candidate_feats = feature_space[:6], feature_space[6:]

# Report the number of features.
print("# static features: {}".format(len(static_feats)))
print("# candidate features: {}\n".format(len(candidate_feats)))


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


# ### 2. 2. Discretization with Random Feature Selection + MDP training (Value Iteration)
#  - Set the same number of bins to the PCA for the comparison

# In[9]:


get_ipython().run_cell_magic('time', '', '## Number of bins for discretization\nn_bins = 7   \n\ncondition = (col_data[\'uniqueval\'] > n_bins)\ncontinuous_feats = col_data[condition].name.tolist()\ncontinuous_feats = [i for i in continuous_feats if i not in static_feats]\n\n# > print the number of selected features\nprint("# continuous features: {}".format(len(continuous_feats)))\n\n# Discretize the continuous features \nfrom sklearn.preprocessing import KBinsDiscretizer\nenc = KBinsDiscretizer(n_bins=n_bins, encode=\'ordinal\')#, strategy=\'uniform\')\n\nnew_data = org_data.copy()\n\n# for col in continuous_feats:\nfor col in candidate_feats:\n    data_reshaped = np.reshape(np.array(org_data[col]), (-1, 1))\n    new_data[col] = enc.fit_transform(data_reshaped)\n\n\n## Random feature selection\nx = []\ny = []\nfeature_sets = {}\ncan_size = len(candidate_feats)\n\n\nfor k in range(1, 9):\n    print("{} selected:".format(k), end=" ")\n    # random select k features from the candidate feature set.\n    chosen_feats = random.sample(candidate_feats, k)\n    # store the chosen feature set in a dictionary\n    feature_sets[k] = chosen_feats\n    print(chosen_feats, end=" ")\n    \n    sample_feats = static_feats + chosen_feats\n    sample_data = new_data[sample_feats]\n    \n    # load data set with selected or extracted discrete features\n    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)\n    \n    # apply Value Iteration to run the MDP\n    try:\n        vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)\n        vi.run()\n\n        # evaluate policy using ECR\n        ecr = calcuate_ECR(start_states, vi.V)\n        print("-> ECR: {}\\n".format(ecr))\n        \n        x.append(k)\n        y.append(ecr)\n        \n    except OverflowError:\n        print("Error occured!\\n")\n        pass')


# ### 2.4 Results

# In[10]:


plt.plot(x, y, 'ro-')
plt.title('[Result] Random Feature Selection')
plt.xlabel('Number of features')
plt.ylabel('ECR')
plt.show()


# ### 2.5. Value Iteration with the best feature set (random feature selection)

# In[11]:


get_ipython().run_cell_magic('time', '', '# feature counts with the highest ECR.\n\nbest_k = y.index(max(y))+1\nprint("Best k (number of features): {}".format(best_k))\nsample_data = data_sets[best_k]\nsample_feats = list(sample_data.columns.values)\n\n# load data set with selected or extracted discrete features\n[start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(sample_data, sample_feats)\n\n# apply Value Iteration to run the MDP\ntry:\n    vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)\n    vi.run()\n    \n    # output policy\n    output_policy(distinct_acts, distinct_states, vi)\n\nexcept OverflowError:\n    print("Error occured!\\n")\n    pass\n\nn_states = n_bins\n\nfor i in range(1, best_k):\n    n_states *= n_bins\n    \nprint("\\nTotal number of states: {}".format(n_states))')


# ### Report
# - Report the description of PCA and any observations from the experiments.
#     * What is the advantages and limitations of PCA?
# - Report the final feature set and the correponding ECR value, and compare it to the best ECR with the random feature selection. 
#     * How to measure the effectiveness and the efficiency of feature selection / policy? 
#     * In this MDP example, which feature selection method is more effective and efficient based on the measurements above? 
# - Report your final policy and the number of PS/WE actions. Include a brief description of the policy.

# ### Discussion
#  - Among Filtered, Wrapper, and Embedded approaches, what type of feature selection is this?
#  - The advatages/ disadvatages of each type of feature selection methods in an MDP framework?
