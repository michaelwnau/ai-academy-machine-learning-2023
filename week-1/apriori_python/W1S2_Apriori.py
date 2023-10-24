#!/usr/bin/env python
# coding: utf-8

# # Week 1 - Session 2: Apriori Algorithm

# In[4]:


from csv import reader
from collections import defaultdict
from itertools import chain, combinations
from optparse import OptionParser
from utils import *

def apriori(itemSetList, minSup, minConf):
    C1ItemSet = getItemSetFromList(itemSetList)
    # Final result global frequent itemset
    globalFreqItemSet = dict()
    # Storing global itemset with support count
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    k = 2

    # Calculating frequent item set
    while(currentLSet):
        # Storing frequent itemset
        globalFreqItemSet[k-1] = currentLSet
        # Self-joining Lk
        candidateSet = getUnion(currentLSet, k)
        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(
            candidateSet, itemSetList, minSup, globalItemSetWithSup)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules

def aprioriFromFile(fname, minSup, minConf):
    C1ItemSet, itemSetList = getFromFile(fname)

    # Final result global frequent itemset
    globalFreqItemSet = dict()
    # Storing global itemset with support count
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    k = 2

    # Calculating frequent item set
    while(currentLSet):
        # Storing frequent itemset
        globalFreqItemSet[k-1] = currentLSet
        # Self-joining Lk
        candidateSet = getUnion(currentLSet, k)
        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(
            candidateSet, itemSetList, minSup, globalItemSetWithSup)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules


# In[5]:


# TID Items
itemSetList = [['A','B'],
               ['A','B','E'],
               ['A','B','D'],
               ['B','E'], 
               ['A','B','D','E'], 
               ['C','D'], 
               ['B','C','D','F'], 
               ['A','D'], 
               ['A','B','D','F'], 
               ['A','B','C']]
freqItemSet, rules = apriori(itemSetList, minSup=0.2, minConf=0.2)



# aprioriFromFile(fname='../dataset/tesco2.csv', minSup=0.2, minConf=0.2)


# aprioriFromFile(fname='../dataset/kaggle.csv', minSup=0.2, minConf=0.2)

