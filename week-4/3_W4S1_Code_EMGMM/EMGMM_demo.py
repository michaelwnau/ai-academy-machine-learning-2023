#!/usr/bin/env python
# coding: utf-8

# ## Week 4 - Session 1 : EM & GMM

# In[ ]:


# This code is released under the MIT license.
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    """ Draw a scatter plot for X associated with gmm result"""
    # gmm: initialized GMM model
    # X: loaded dataset
    
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=15, alpha=0.5, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        
def plot_scatter(X):
    """ Draw a scatter plot for X """
    # X: loaded dataset
    
    plt.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, cmap='viridis')


# ### Question 1 >>
# * Load and plot the data: "EMGMM_dataset.npy"
# * What do you observe?

# In[ ]:


data = np.load("EMGMM_dataset.npy")
print(data.shape)
plot_scatter(data)


# ### Question 2 >>
# * Change the parameters, compare the results
# * Please refer to the parameters of GMM in this link: 
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

# In[ ]:


gmm = GaussianMixture(n_components=3,
                      covariance_type='full',
                      random_state=42,
                      init_params='random',
                      verbose=0)
plot_gmm(gmm, data)


# ### Question 3 >>
# * Plot the GMM result with different number of iterations.
# * Cahnge the iteration: 7 -> 15 -> 25

# In[ ]:


gmm = GaussianMixture(n_components=3,
                      covariance_type='full',
                      random_state=42,
                      init_params='random',
                      verbose=0,
                      max_iter=7)
plot_gmm(gmm, data)


# ### Question 4: Explore the MIMIC-III data
# * Apply the methods to the imputed MIMIC-III data set with any two numerical features.
# * What happens if the data for each cluster do not follow a Gaussian distribution?
# * What happens if the data of two clusters are heavily overlapped?

# In[ ]:


import pandas as pd
shock = pd.read_csv("../s2_Code_Clustering/mimic_shock.csv", header=0)
nonshock = pd.read_csv("../s2_Code_Clustering/mimic_nonshock.csv", header=0)
feat = ['SystolicBP','HeartRate']

shockVids = shock.VisitIdentifier.unique().tolist()[:100]
nonshockVids = nonshock.VisitIdentifier.unique().tolist()[:100]

shockSelected = shock.loc[shock.VisitIdentifier.isin(shockVids), feat]
nonshockSelected = nonshock.loc[nonshock.VisitIdentifier.isin(nonshockVids),feat]

shock_data = shockSelected.values
nonshock_data = nonshockSelected.values
mimic_data = np.concatenate([shock_data, nonshock_data])

plot_scatter(mimic_data)


# In[ ]:


plot_scatter(shock_data)


# In[ ]:


plot_scatter(nonshock_data)


# In[ ]:


# GMM
gmm = GaussianMixture(n_components=2,
                      n_init = 1,  # default number of init = 1
                      covariance_type='full',
                      random_state=2,
                      init_params='random',
                      verbose=0)

print("* GMM result")
plot_gmm(gmm, mimic_data)


# ### Question 5: You may play with your own data or another dataset you generate.
# * You may artificially gerenate data as follows:

# In[ ]:


from sklearn.datasets import make_blobs   # scikit-learn version = 1.0.2  
# if you fail to import the above, try this older version. 
#from sklearn.datasets.samples_generator import make_blobs 

X, y_true = make_blobs(n_samples=400, centers=5,
                       cluster_std=0.9, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
rng = np.random.RandomState(7)
X1 = np.dot(X[:200], rng.randn(2, 2))  # make the first half of data skewed with random vectors
X2 = np.dot(X[200:], rng.randn(2, 2)) # make the second half of data skewed with random vectors
new_data = np.concatenate([X1, X2])
plot_scatter(new_data)


# In[ ]:


# GMM Clustering
gmm = GaussianMixture(n_components=4,
                      n_init = 10,  # default number of init = 1
                      covariance_type='full',
                      random_state=2,
                      init_params='random',
                      verbose=0)

print("* GMM result")
plot_gmm(gmm, new_data)

