#!/usr/bin/env python
# coding: utf-8

# ## Week 8 - Session 1: Hidden Markov Model (HMM)
#  - Explore an intelligent tutoring system called Deep Thought.
#  - Deep Thought takes two actions, providing 1) Problem Solving (PS) and 2) Work Example (WE), based on the students' state for their best learning gain.

# In[1]:


__author__ = 'yemao'
from HMM import hmm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, mean_squared_error


# ### Data

# In[2]:


np.random.seed(1000)
folder = "assignment_data/"             # Specify your folder here
qlg_y = np.load(folder + "y_qlg.npy")   # target label for learning gain
post_y = np.load(folder + "post_test.npy", allow_pickle=True)   # target value for post-test score
qlg_test_actual, post_test_actual = [], []


# ### Training (5-CV)

# In[3]:


get_ipython().run_cell_magic('time', '', 'kf = KFold(n_splits=5, shuffle=True)\nqlg_train_pred, post_train_pred = [], []\nqlg_train_actual, post_train_actual = [], []\nqlg_test_pred, post_test_pred = [], []\n\nnumkc = 7      # Q2-1. change number of kc for different data set\n\nfor train_index, test_index in kf.split(qlg_y):\n    print("======================================================")\n    qlg_y_train, qlg_y_test = qlg_y[train_index], qlg_y[test_index]\n    post_y_train, post_y_test = post_y[train_index], post_y[test_index]\n\n    # symbols here refers to two different observations:\n    # 1: correct, 0: incorrect\n    symbols = [[\'0\', \'1\']]\n    \n    #------------------------------------------\n    # Q2-2. Explore different parameters: \n    #   1) Pi : Initial staste prob.\n    #   2) T : State transition prob.\n    # -----------------------------------------\n\n    #h = hmm(2, Pi=np.array([0.5, 0.5]), T=np.array([[0.86, 0.14], [0.09, 0.91]]), obs_symbols=symbols)\n    \n    \n    nlg_train = [[] for x in range(numkc)]\n    nlg_test = [[] for x in range(numkc)]\n    \n    for i in range(numkc):\n        print("-----------------------------------------")\n        print(" KC: {}".format(i))\n        h = hmm(2, Pi=np.array([0.5, 0.5]), T=np.array([[0.86, 0.14], [0.09, 0.91]]), obs_symbols=symbols)\n        \n        X = np.load(folder + "perf_kc" + str(i+1) + ".npy", allow_pickle=True)\n        X_train, X_test = X[train_index], X[test_index]\n\n        train = [each for each in X_train if each]\n        test = [each for each in X_test if each]\n        \n        \n        if train and test:\n            h.baum_welch(train, debug=False)        # Baum-Welch algorithm : training part\n        \n        \n        nlg_train[i].extend(h.predict_nlg(X_train))\n        nlg_test[i].extend(h.predict_nlg(X_test))\n\n    print(len(nlg_train), len(nlg_train[0]))\n    nlg_train = np.transpose(nlg_train)\n    nlg_test = np.transpose(nlg_test)\n\n    nlg_train = pd.DataFrame(nlg_train).fillna(value=0)\n    nlg_test = pd.DataFrame(nlg_test).fillna(value=0)\n\n    # ---------------------------------------------------------\n    # logistic regression for learning gain prediction\n    logreg = LogisticRegression()                   \n    logreg.fit(nlg_train, qlg_y_train)\n    predict = logreg.predict(nlg_train)\n    qlg_train_pred.extend([each for each in predict])\n    qlg_train_actual.extend(qlg_y_train)\n\n    predict = logreg.predict(nlg_test)\n    # print (logreg.predict_proba(nlg_test))\n    qlg_test_pred.extend([each for each in predict])\n    qlg_test_actual.extend(qlg_y_test)\n\n\n    lg = LinearRegression()              # linear regression for post-test scores prediction\n    lg.fit(nlg_train, post_y_train)\n    predict = lg.predict(nlg_train)\n    post_train_pred.extend([each for each in predict])\n    post_train_actual.extend(post_y_train)\n\n    predict = lg.predict(nlg_test)\n    post_test_pred.extend([each for each in predict])\n    post_test_actual.extend(post_y_test)')


# ### Results

# In[4]:


# test code data ###
# qlg_test_pred, post_test_pred = [0]*len(qlg_y), [0.5]*len(post_y)
# qlg_test_actual, post_test_actual = qlg_y, post_y
print( " ")
print ("<<<<<<< student learning gain")
print ("Training accuracy:" + str(accuracy_score(qlg_train_actual, qlg_train_pred)))
print ("Accuracy: " + str(accuracy_score(qlg_test_actual, qlg_test_pred)))


# flip P and N here because we care about the low learning gain group: qlg = 0
qlg_test_actual = [1 if each == 0 else 0 for each in qlg_test_actual]       
qlg_test_pred = [1 if each == 0 else 0 for each in qlg_test_pred]


print( "f1_score: " + str(f1_score(qlg_test_actual, qlg_test_pred)))
print ("Recall: " + str(recall_score(qlg_test_actual, qlg_test_pred)))
print ("AUC: " + str(roc_auc_score(qlg_test_actual, qlg_test_pred)))
print ("Confusion Matrix: ")
print (confusion_matrix(qlg_test_actual, qlg_test_pred))
print (" ")
print ("<<<<<<< student modeling")
print ("Training MSE: ", mean_squared_error(post_train_actual, post_train_pred))
print ("Test MSE: ", mean_squared_error(post_test_actual, post_test_pred))


# ### Report your observation from
#  - the HMM model parameters after training as compared to the initial parameters (select one KC in the first fold of 5-CV). 
#  - changing the initial parameters of the HMM model.
#  - comparing parameters of HMMs trained for different KCs.
