#!/usr/bin/env python
# coding: utf-8

# ## Week 4 - Session 2 : EM & GMM vs. K-means

# In[ ]:


# This code is released under the MIT license.
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

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
        
def plot_gmm(gmm, X, label=True, ax=None, xlabel='', ylabel=''):
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
        
    ax.set_xlabel(xlabel, fontsize=16) # added
    ax.set_ylabel(ylabel, fontsize=16)        

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None, xlabel='', ylabel=''):
    """ Draw a scatter plot for X associated with gmm result"""
    # kmeans: initialized K-Means model
    # X: loaded dataset
    
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=15, alpha = 0.5, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        
    ax.set_xlabel(xlabel, fontsize=16) # added
    ax.set_ylabel(ylabel, fontsize=16)        
        
        
def plot_scatter(X, xlabel='', ylabel=''):
    """ Draw a scatter plot for X """
    # X: loaded dataset
    
    plt.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, cmap='viridis')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16) 
    
    
# Train GMM and show the learned parameters and predicted labels
def train_gmm(gmm, data):
    labels = gmm.fit(data).predict(data)
    gmm.fit(data).predict(data)
    print("n_iter_: {} \t(converged_: {})\nwegiths_: {}\nmeans_: {}\ncovariances_:\n{}".format(gmm.n_iter_, 
                                     gmm.converged_, gmm.weights_.round(3), gmm.means_.round(3), 
                                     gmm.covariances_.round(3)))
    print("first 20 labels: {} ...".format(labels[:20])) 

# Train K-means and show the learned parameters and predicted labels    
def train_kmeans(kmeans, data):
    labels = kmeans.fit_predict(data)
    print("n_iter_: {} \ncluster_centers_:\n{}".format(kmeans.n_iter_, kmeans.cluster_centers_.round(3)))
    print("first 20 labels: {} ...".format(labels[:20]))     


# ### Question 1 >> 
# * Load and plot the data.
# * What do you observe?

# In[ ]:


# 1. Load dataset - "EMGMM_Kmeans_dataset.npy"
data = np.load("EMGMM_Kmeans_dataset.npy")

# 1-1. Print the shape of data
print("data.shape = {}".format(data.shape))

# 1-2. Draw a scatter plot - use the 'plot_scatter' function.
plot_scatter(data)


# ### Question 2 >>
# * Train GMMs and K-means, check the predicted labels and the learned parameters of each model: 
#   - GMM: weights of mixtures, means, covariance 
#   - K-means: centers
# * Please refer to the following links for the parameters and attributes of each function:  
#  - GMM: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
#     - GMM covariance type: https://i.stack.imgur.com/0zLpe.png
#  - K-means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# In[ ]:


# GMM Clustering
gmm = GaussianMixture(n_components=4,
                      n_init = 1,  # default number of init = 1
                      covariance_type='full',
                      random_state=2,
                      init_params='random',
                      verbose=0)
print("* GMM result")
plot_gmm(gmm, data)
train_gmm(gmm, data)


# In[ ]:


# plot_gmm does not support for covariance_type : shperical or tied 
# Please use train_gmm without plotting them to check the results
gmm = GaussianMixture(n_components=4,
                      n_init = 1,  # default number of init = 1
                      covariance_type='spherical', 
                      random_state=2,
                      init_params='random',
                      verbose=0)
print("* GMM result")
train_gmm(gmm, data)


# In[ ]:


# K-Means Clustering
kmeans = KMeans(n_clusters=4,
                n_init = 10, # default number of init = 10 
                random_state=2)
print("* K-Means result")
plot_kmeans(kmeans, data)
train_kmeans(kmeans, data)


# ### Question 3 >> 
# * Briefly describe how the clusters are different (or the same) and why, then choose one
# method.

# * We can observe from the two plots that GMM and K-Means have different shape of clusters annd membership of datapoints. Specifically, GMM result shows a clearer shape of clusters compared to K-means by capturing ellipse shape of data distribution. For this type of data, I would choose GMM over K-means.

# ### Question 4: Explore the MIMIC-III data
# * Apply GMM and K-means to the imputed MIMIC-III data set with any two numerical features.
# * How differently GMM and K-means work with MIMIC-III data?

# In[ ]:


import pandas as pd
shock = pd.read_csv("mimic_shock.csv", header=0)
nonshock = pd.read_csv("mimic_nonshock.csv", header=0)
print("available features: {}".format(shock.columns[2:21].values))
feat = ['RespiratoryRate','MAP']

shockVids = shock.VisitIdentifier.unique().tolist()[:100]
nonshockVids = nonshock.VisitIdentifier.unique().tolist()[:100]

shockSelected = shock.loc[shock.VisitIdentifier.isin(shockVids), feat]
nonshockSelected = nonshock.loc[nonshock.VisitIdentifier.isin(nonshockVids),feat]

shock_data = shockSelected.values
nonshock_data = nonshockSelected.values
mimic_data = np.concatenate([shock_data, nonshock_data])

# Use mimic_data to train GMM, assuming we do not know the labels of shock / non-shock classes
print("minic_data: the whole data without labels ")
plot_scatter(mimic_data, xlabel=feat[0], ylabel=feat[1])


# In[ ]:


# shock class
plot_scatter(shock_data, xlabel=feat[0], ylabel=feat[1])


# In[ ]:


# non-shock class
plot_scatter(nonshock_data, xlabel=feat[0], ylabel=feat[1])


# In[ ]:


# GMM
gmm = GaussianMixture(n_components=2,
                      n_init = 1,  # default number of init = 1
                      covariance_type='full',
                      random_state=2,
                      init_params='random',
                      verbose=0)
print("* GMM result")
plot_gmm(gmm, mimic_data, xlabel=feat[0], ylabel=feat[1])
train_gmm(gmm, mimic_data)


# In[ ]:


# K-Means Clustering
kmeans = KMeans(n_clusters=2,
                n_init = 10, # default number of init = 10 
                random_state=2)

print("* K-Means result")
plot_kmeans(kmeans, mimic_data,xlabel=feat[0], ylabel=feat[1])
train_kmeans(kmeans, mimic_data)


# ### Question 5: You may play with your own data or another dataset you generate
# * You may artificially gerenate data with the below code and train them with GMM and K-means.
# * Change the number of clusters (n_cluster). What happens if we have a different assumption of the number of clusters (n_components in GaussianMixture/ n_cluster in KMeans) from the actual latent number of clusters (centers in make_blobs)?
# * Explore other hyper-parameters
# * What are your findings, comparing GMM with K-means?

# In[ ]:


from sklearn.datasets import make_blobs   # scikit-learn version = 1.0.2  
# if you fail to import the above, try this older version. 
#from sklearn.datasets.samples_generator import make_blobs 

X, y_true = make_blobs(n_samples=400, centers=7,
                       cluster_std=0.99, random_state=1)
X = X[:, ::-1] # flip axes for better plotting
rng = np.random.RandomState(7)
X1 = np.dot(X[:200], rng.randn(2, 2))  # make the first half of data skewed with random vectors
X2 = np.dot(X[200:], rng.randn(2, 2)) # make the second half of data skewed with random vectors
new_data = np.concatenate([X1, X2])
plot_scatter(new_data)


# In[ ]:


# GMM Clustering
gmm = GaussianMixture(n_components=4,  #### Explore different n_components 
                      n_init = 10,  # default number of init = 1
                      covariance_type='full',
                      random_state=2,
                      init_params='random',
                      verbose=0)

print("* GMM result")
plot_gmm(gmm, new_data)
train_gmm(gmm, new_data)


# In[ ]:


# K-Means Clustering
kmeans = KMeans(n_clusters=4,  #### Explore different n_clusters 
                n_init = 10, # default number of init = 10 
                random_state=2)

print("* K-Means result")
plot_kmeans(kmeans, new_data)
train_kmeans(kmeans, new_data)

