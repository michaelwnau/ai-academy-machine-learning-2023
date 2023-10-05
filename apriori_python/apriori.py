"""
This code snippet is an implementation of the Apriori algorithm, a popular algorithm for frequent itemset mining and association rule learning over transactional databases. It takes a CSV file as input and calculates the frequent itemsets and association rules based on the minimum support and confidence thresholds provided.

Example Usage:
python code.py -f input.csv -s 0.5 -c 0.5

Inputs:
- inputFile: a CSV file containing transaction data
- minSup: the minimum support threshold for frequent itemsets (default: 0.5)
- minConf: the minimum confidence threshold for association rules (default: 0.5)

Outputs:
- globalFreqItemSet: a dictionary containing the frequent itemsets, where the key is the size of the itemset and the value is a set of itemsets
- rules: a list of association rules, where each rule is represented as a tuple (antecedent, consequent, confidence)

Code Analysis:
1. The code snippet parses the command line arguments to get the input file, minimum support, and minimum confidence values.
2. It calls the 'aprioriFromFile' function, passing the input file, minimum support, and minimum confidence as arguments.
3. The 'aprioriFromFile' function reads the input file and converts it into a list of itemsets.
4. It initializes the global frequent itemset and global itemset with support count.
5. It calculates the L1 itemset (frequent itemset of size 1) by filtering out itemsets with support below the minimum support threshold.
6. It iteratively calculates the frequent itemsets of size k (k > 1) by self-joining the previous frequent itemsets, pruning supersets, and counting the support of the candidate itemsets.
7. It generates the association rules based on the frequent itemsets and support counts.
8. The frequent itemsets and association rules are sorted based on confidence.
9. The global frequent itemsets and association rules are returned as the output.

"""

from csv import reader
from collections import defaultdict
from itertools import chain, combinations
from optparse import OptionParser
from apriori_python.utils import *

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

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='inputFile',
                         help='CSV filename',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minSup',
                         help='Min support (float)',
                         default=0.5,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minConf',
                         help='Min confidence (float)',
                         default=0.5,
                         type='float')

    (options, args) = optparser.parse_args()

    freqItemSet, rules = aprioriFromFile(options.inputFile, options.minSup, options.minConf)
