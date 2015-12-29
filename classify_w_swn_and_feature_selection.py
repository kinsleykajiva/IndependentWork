import nltk, re, pprint
import math
from nltk import word_tokenize

# SentiWordNet Feature Selection
#import numpy as np

# modified from https://de.dariah.eu/tatom/feature_selection.html
# and https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/

from nltk.corpus import sentiwordnet as swn

class FeatureSelection(object):

    def __init__(self):
        self.wdict = {}
        self.dcount = 0
        self.s = 0
        self.n = 0
        labels = []

    def parse(self, doc, sentiment):
        tokens = word_tokenize(doc)
        words = [w.lower() for w in tokens]
        pos_tagged_words = nltk.pos_tag(words)
        adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
        set_adjectives = set(adjectives)
        s = 0
        n = 0
        if sentiment == "Satisfied":
            s = 1
            self.s += 1
        else:
            n = 1
            self.n += 1
        for adj in set_adjectives:
            if adj in self.wdict:
                (sat,notsat) = self.wdict[adj]
                self.wdict[adj] = ((sat+s),(notsat+n))
            else:
                self.wdict[adj] = (s,n)
        self.dcount += 1

    # top 30 chi-squared values
    def ret_keys(self):
        k = self.keys
        return k[-20:]

    def rem_nonevaluative(self):
        for key, value in self.wdict.items():
            scores = swn.senti_synsets(key, 'a')
            if not list(scores):
                del self.wdict[key]
                continue
            score = list(scores)[0]
            objectivity = score.obj_score()
            if objectivity >= 0.5:
                del self.wdict[key]
        
    def calc(self):
        keys = []
        for key, value in self.wdict.items():
            (s,n) = value
            t_expected = float(self.s)*(float(s+n)/self.dcount)
            t_observed = s
            diff = math.pow((float(t_observed) - t_expected),2)
            chi_s = float(diff)/t_expected
            keys.append((key,chi_s))
        self.keys = [k for k, chi_s in sorted(keys, key=lambda k: k[1])]

fs = FeatureSelection()
# Modified from chapter 6 of NLTK

#f = open('_SOC_.txt', 'r')
#adj_features = []
# parse each review                                                            
#for line in f.readlines():
#    fs.parse(line)
#fs.calc()
#set_of_adjectives_features = fs.ret_keys()
#print(set_of_adjectives_features)
#print(len(set_of_adjectives_features))

# Feature vectorization
def review_features(review):
    tokens = word_tokenize(review)
    words = [w.lower() for w in tokens]
    # TODO: modify choice of pos tagger
    pos_tagged_words = nltk.pos_tag(words)
    adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
    set_adjectives = set(adjectives)
    features = {}
    my_keys = fs.ret_keys()
    for adj in my_keys:
        features['contains({})'.format(adj)] = adj in set_adjectives
    return features

featuresets = []

from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

import csv

#feature_selector = Chi_Squared()   

# read in gold standard
with open('batch_results_stem.csv','rU') as csvfile:
    reader = csv.DictReader(csvfile)
    for row1,row2,row3 in grouper(reader, 3):
        review = row1['Review']
        sentiment1 = row1['Answer']
        sentiment2 = row2['Answer']
        sentiment3 = row3['Answer']
        sentiment = "Not Satisfied"
        # test for agreement
        if (sentiment1 == "Satisfied" and (sentiment2 == "Satisfied" or sentiment3 == "Satisfied")) or (sentiment2 == "Satisfied" and sentiment3 == "Satisfied"):
            sentiment = "Satisfied"
        p = (review,sentiment)
        featuresets.append(p)
        fs.parse(review,sentiment)

fs.rem_nonevaluative()
fs.calc()

final_featuresets = []

for review, sentiment in featuresets:
    final_featuresets.append((review_features(review),sentiment))

dev_test_set = [{'contains(easy)': True},{'contains(hard)': True},{'contains(funny)': True},{'contains(useful)': True},{'contains(interesting)': True}]

#feature_selector.build_count_matrix()
#features = feature_selector.features()

#print featuresets[0:1]

#final_featuresets = []

#print len(my_keys)

# update to only include Chi-Squared selected features
#for review in featuresets:
  #  review_features, sentiment = review
   # new_features = {}
   # for adj in my_keys:
    #    if 'contains({})'.format(adj) in review_features:
    #        new_features['contains({})'.format(adj)] = review_features['contains({})'.format(adj)]
            #print review_features['contains({})'.format(adj)]
    #    else:
    #        new_features['contains({})'.format(adj)] = False
   # final_featuresets.append((new_features,sentiment))
    #print count

#print final_featuresets[0:1]

# divide featuresets
len_of_featuresets = len(final_featuresets)
three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)

#TRIAL 1
train_set, test_set, devtest_set = final_featuresets[three_quarters:], final_featuresets[:one_quarter], final_featuresets[one_quarter:three_quarters]

# classify
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Error Analysis

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative

for review_features,sentiment in test_set:
    if sentiment == "Satisfied":
        if classifier.classify(review_features) == "Satisfied":
            tp += 1
        else:
            fn += 1
    else:
        if classifier.classify(review_features) == "Satisfied":
            fp +=1


precision = float(tp)/(tp+fp) # cast to float to avoid integer division
recall = float(tp)/(tp+fn)
print precision
print recall
fmeasure = 2.0*float(precision*recall)/(precision+recall)
print fmeasure

for features in dev_test_set:
    guess = classifier.classify(features)
    print guess

# TRIAL 2
train_set, test_set, devtest_set = final_featuresets[-three_quarters:], final_featuresets[one_quarter:], final_featuresets[one_quarter:three_quarters]

# classify
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Error Analysis

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative

for review_features,sentiment in test_set:
    if sentiment == "Satisfied":
        if classifier.classify(review_features) == "Satisfied":
            tp += 1
        else:
            fn += 1
    else:
        if classifier.classify(review_features) == "Satisfied":
            fp +=1


precision = float(tp)/(tp+fp) # cast to float to avoid integer division
recall = float(tp)/(tp+fn)
print precision
print recall
fmeasure = 2.0*float(precision*recall)/(precision+recall)
print fmeasure

for features in dev_test_set:
    guess = classifier.classify(features)
    print guess

# TRIAL 3
train_set, test_set = final_featuresets[one_quarter:]+final_featuresets[(one_quarter*2):len_of_featuresets], final_featuresets[one_quarter:(2*one_quarter)]

# classify
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Error Analysis

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative

for review_features,sentiment in test_set:
    if sentiment == "Satisfied":
        if classifier.classify(review_features) == "Satisfied":
            tp += 1
        else:
            fn += 1
    else:
        if classifier.classify(review_features) == "Satisfied":
            fp +=1


precision = float(tp)/(tp+fp) # cast to float to avoid integer division
recall = float(tp)/(tp+fn)
print precision
print recall
fmeasure = 2.0*float(precision*recall)/(precision+recall)
print fmeasure

for features in dev_test_set:
    guess = classifier.classify(features)
    print guess

# TRIAL 4
train_set, test_set, devtest_set = final_featuresets[(one_quarter*2):]+final_featuresets[-one_quarter:], final_featuresets[(2*one_quarter):(3*one_quarter)], final_featuresets[one_quarter:three_quarters]

# classify
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Error Analysis

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative

for review_features,sentiment in test_set:
    if sentiment == "Satisfied":
        if classifier.classify(review_features) == "Satisfied":
            tp += 1
        else:
            fn += 1
    else:
        if classifier.classify(review_features) == "Satisfied":
            fp +=1


precision = float(tp)/(tp+fp) # cast to float to avoid integer division
recall = float(tp)/(tp+fn)
print precision
print recall
fmeasure = 2.0*float(precision*recall)/(precision+recall)
print fmeasure

for features in dev_test_set:
    guess = classifier.classify(features)
    print guess
