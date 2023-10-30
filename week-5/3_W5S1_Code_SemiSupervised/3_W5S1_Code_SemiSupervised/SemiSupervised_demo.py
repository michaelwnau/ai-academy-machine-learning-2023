#!/usr/bin/env python
# coding: utf-8

# ## Week 5 - Session 1: Semi-Supervised Learning

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import random 
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from frameworks.SelfLearning import *
from sklearn.semi_supervised import LabelPropagation


# ### Data
#  * https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

# * There are 6 numerical and 8 categorical attributes. The labels have been changed for the convenience of the statistical algorithms. For example, attribute 4 originally had 3 labels p,g,gg and these have been changed to labels 1,2,3.
# 
#         A1: 0,1 CATEGORICAL (formerly: a,b)
#         A2: continuous.
#         A3: continuous.
#         A4: 1,2,3 CATEGORICAL (formerly: p,g,gg)
#         A5: 1, 2,3,4,5, 6,7,8,9,10,11,12,13,14 CATEGORICAL (formerly: ff,d,i,k,j,aa,m,c,w, e, q, r,cc, x)
#         A6: 1, 2,3, 4,5,6,7,8,9 CATEGORICAL (formerly: ff,dd,j,bb,v,n,o,h,z)
#         A7: continuous.
#         A8: 1, 0 CATEGORICAL (formerly: t, f)
#         A9: 1, 0 CATEGORICAL (formerly: t, f)
#         A10: continuous.
#         A11: 1, 0 CATEGORICAL (formerly t, f)
#         A12: 1, 2, 3 CATEGORICAL (formerly: s, g, p)
#         A13: continuous.
#         A14: continuous.
#         A15: 1,2 class attribute (formerly: +,-)

# In[2]:


# load data and preprocessing
def load_data():
    
    df = pd.read_csv("australian_credit.csv", header=None, sep='\t')
    df.columns = ['A'+str(i+1) for i in range(14)] + ['label']
    cat_feat = ['A4', 'A5', 'A6', 'A12']
    num_feat = [f for f in df.columns[:-1] if f not in cat_feat]
    
    # One-hot encoding for (multi-) categorical data
    df = pd.get_dummies(df, columns = cat_feat)
    df = df[[c for c in df.columns if c not in ['label']] + ['label']]
    # Normalization for numerical data
    
    x = df[num_feat].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df.loc[:, num_feat] = x_scaled

    return df

# Split data into the training and test data with the given split_ratio, 
# and set labeled/ unlabeled data with the given percent
def getLabeledData(df, split_ratio, percent): 
    # Data split with Stratified random sampling
    posdf = df[df.label==1].reset_index(drop=True)
    negdf = df[df.label==0].reset_index(drop=True)
    
    #posdf = posdf.sample(frac = 1)
    #negdf = negdf.sample(frac = 1)
    
    poslen = len(posdf)
    neglen = len(negdf)
    print("pos: {}, neg:{}".format(posdf.shape, negdf.shape))

    # 1. split data into train and test
    numPosTrain = int(poslen * split_ratio)
    numNegTrain = int(neglen * split_ratio)
    posTrain = posdf.iloc[:numPosTrain, :]
    negTrain = negdf.iloc[:numNegTrain, :]
    posTest = posdf.iloc[numPosTrain:, :]
    negTest = negdf.iloc[numNegTrain:, :]
    
    # 2. extract labeled data from the training data into labeld and unlabeled
    posLabeled = posTrain.iloc[: int(numPosTrain * percent), :]
    negLabeled = negTrain.iloc[: int(numNegTrain * percent), :] 

    posUnlabeled = posTrain.iloc[int(numPosTrain * percent):, :]
    negUnlabeled = negTrain.iloc[int(numNegTrain * percent):, :] 
    
    labeled = pd.concat([posLabeled, negLabeled], axis=0, sort=False) 
    unlabeled = pd.concat([posUnlabeled, negUnlabeled], axis=0, sort=False) 
    
    X_labeled = labeled.iloc[:,:-1].values
    y_labeled = labeled.iloc[:,-1].values
    X_unlabeled = unlabeled.iloc[:,:-1].values
    y_unlabeled = unlabeled.iloc[:,-1].values

    X_train_total = np.concatenate((np.array(X_labeled), np.array(X_unlabeled)), axis=0)
    # For unlabeled data, set the labels to -1
    y_train_total = np.concatenate((np.array(y_labeled), np.array([-1]*y_unlabeled.shape[0])), axis=0)
    
    testdf  = pd.concat([posTest, negTest], axis=0, sort=False)
    X_test = testdf.iloc[:,:-1].values
    y_test = testdf.iloc[:,-1].values     
    
    return X_labeled, y_labeled, X_train_total, y_train_total, X_test, y_test


# ### Model training

# In[3]:


def supervised_svm(X_labeled, y_labeled, X_test, y_test):
    model = svm.SVC(probability=True, gamma = 0.1)
    model.fit(X_labeled, y_labeled)
    return evaluate(model, X_test, y_test, "Supervised SVM")


def self_learning(base_model, X_train_total, y_train_total, X_test, y_test):
    ss_model = SelfLearningModel(base_model, prob_threshold=0.6, max_iter=300)
    ss_model.fit(X_train_total, y_train_total)
    return evaluate(ss_model, X_test, y_test, "Self learning")


def label_propagation(X_train_total, y_train_total, X_test, y_test):
    model = LabelPropagation(max_iter=5000)
    model.fit(X_train_total, y_train_total)
    return evaluate(model, X_test, y_test, "Label propag.")


# ### Looking at the above code answer the following:
# 
# 1. Describe SVM in a couple sentences.
# 2. Describe self learning in a couple sentences.
# 3. Descirbe label probagation in a couple sentences.

# ### Evaluation

# In[4]:


def evaluate(model, X_test, y_test, category=''): 
    
    test_predicted = model.predict(X_test)
    test_predicted_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, test_predicted)
    
    # When all the data are classified into one cluster.
    if np.mean(test_predicted)==0 or np.mean(test_predicted)==1:   
        precision = 0
    else:
        precision = precision_score(y_test, test_predicted)
    
    recall = recall_score(y_test, test_predicted)
    f_measure = f1_score(y_test, test_predicted)
    tsp_prob = []
    for each in test_predicted_prob:
        tsp_prob.append(each[1])

    fpr, tpr, thresholds = metrics.roc_curve(y_test, tsp_prob, pos_label=1)
    auc_roc = metrics.auc(fpr, tpr)

    print ("{}\tf: {:.3f}, recall:{:.3f}\tprec:{:.3f}\tacc:{:.3f}\tauc:{:.3f}".
          format(category, f_measure, recall, precision, accuracy, auc_roc))

    return [accuracy, precision, recall, f_measure, auc_roc]


# ### Questions:
# 
# 1. Which measure of evaluation would be most appropriate for the credit database?  
# 2. Which measure would be inappropriate?

# ### Compare Semi-Supervised Learning with Supervised Learning

# In[5]:


def compare_methods(fold, df, labeled_percent):
    
    for p in labeled_percent:
        
        X_labeled, y_labeled, X_train_total, y_train_total, X_test, y_test = getLabeledData(df, split_ratio, p)
       
        print("\n**** LABELED PERCENTAGE: {}".format(p*100))        
        
        print("X_labeled: {}\tX_train_total: {}\tX_test: {}\ny_labeled: {}  \ty_train_total: {}\t\ty_test: {}".format(
            X_labeled.shape, X_train_total.shape, X_test.shape, y_labeled.shape, y_train_total.shape, y_test.shape))
        print("\n --- SVM")
        base_model = sklearn.svm.SVC(probability=True, gamma = 0.1)
        eval_res = self_learning(base_model, X_train_total, y_train_total, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'SVM', 'SelfLearn'] + eval_res
        
        acc, prec, rec, f_measure, auc = supervised_svm(X_labeled, y_labeled, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'SVM','Supervised'] + [acc, prec, rec, f_measure, auc]
        
        print("\n --- Decision Tree")
        base_model = DecisionTreeClassifier()
        eval_res = self_learning(base_model, X_train_total, y_train_total, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'DT', 'SelfLearn'] + eval_res
        
        base_model = DecisionTreeClassifier()
        base_model.fit(X_labeled, y_labeled)
        eval_res = evaluate(base_model, X_test, y_test, 'Supervised')
        res.loc[len(res)] = [fold, p, 'DT', 'Supervised'] + eval_res

        print("\n --- Logistic Regression")
        base_model = linear_model.LogisticRegression()
        eval_res = self_learning(base_model, X_train_total, y_train_total, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'LR', 'SelfLearn'] + eval_res
        
        base_model = linear_model.LogisticRegression()
        base_model.fit(X_labeled, y_labeled)
        eval_res = evaluate(base_model, X_test, y_test, 'Supervised')
        res.loc[len(res)] = [fold, p, 'LR','Supervised'] + eval_res

        print("\n --- Naive Bayes")
        base_model = GaussianNB()
        eval_res = self_learning(base_model, X_train_total, y_train_total, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'NB', 'SelfLearn'] + eval_res
    
        base_model.fit(X_labeled, y_labeled)
        eval_res = evaluate(base_model, X_test, y_test, "Supervised")
        res.loc[len(res)] = [fold, p, 'NB', 'Supervised'] + eval_res
        
        print("\n --- Label Propagation")
        eval_res = label_propagation(X_train_total, y_train_total, X_test, y_test)
        res.loc[len(res)] = [fold, p, 'Label_Prop', np.nan] + eval_res    

    return res   


# In[6]:


if __name__ == "__main__":
    split_ratio = 0.7 
    labeled_percent = [0.01, 0.02, 0.1, 0.15]
    models = ['SVM', 'DT', 'LR', 'NB', 'Label_Prop']
    approaches = ['SelfLearn', 'Supervised']
    random.seed(100)
    
    # load and preprocess data 
    df = load_data()
    
    # train and evaluate
    res = pd.DataFrame(columns=['fold', 'Labeled', 'Model', 'Approach', 'Accuracy', 'Precision', 
                                'Recall', 'F-measure', 'AUC'])    
    for i in range(20):
        res = compare_methods(i, df, labeled_percent)
    

    # get the average of evaluation metrics
    repeatRes = pd.DataFrame(columns=res.columns[1:])
    
    for l in labeled_percent:
        for m in models:
            if m != 'Label_Prop':
                for a in approaches:
                    tmp = res[(res.Labeled==l)& (res.Model==m) & (res.Approach==a)][res.columns[4:]].mean().values.tolist()
                    repeatRes.loc[len(repeatRes)] = [l, m, a]+ tmp        
            else:
                tmp = res[(res.Labeled==l)& (res.Model==m)][res.columns[4:]].mean().values.tolist()
                repeatRes.loc[len(repeatRes)] = [l, m, '']+ tmp        


# In[7]:


# Overall comparison
repeatRes.round(3)


# ### Look at the above table and answer the following questions:
# 
# 1. Which method performed best under which circumstances?
# 2. What metrics did you use to make the above determination?

# In[8]:


repeatRes[(repeatRes.Model=='SVM')].round(3)


# In[9]:


repeatRes[(repeatRes.Model=='LR')].round(3)


# In[10]:


repeatRes[(repeatRes.Model=='DT')].round(3)


# In[11]:


repeatRes[(repeatRes.Model=='NB')].round(3)


# In[12]:


repeatRes[(repeatRes.Model=='Label_Prop')].round(3)


# ## After running the above cells, answer these questions:
# 1. Compare the performance of these semi-supervised models against the supervised base classifiers such as: SVM, NB, DT, and LR trained on the same set of labeled data.
# 2. How do the results change as the percent of labeled data increases? 
# 3. Why do you think such results are produced?

# In[ ]:




