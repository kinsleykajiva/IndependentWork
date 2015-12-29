import nltk, re, pprint
from nltk import word_tokenize
import math

# Chi-Square Feature Selection
#import numpy as np

# modified from https://de.dariah.eu/tatom/feature_selection.html
# and https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/                                    
class Chi_Squared(object):

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
        return k[-20:] # largest values (dependence) 
        
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

fs = Chi_Squared()
#set_of_adjectives_features = []
#f = open('_HUM_.txt', 'r')
#for review in f.readlines():
#    tokens = word_tokenize(review)
 ##   pos_tagged_words = nltk.pos_tag(words)
  #  adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
  #  set_of_adjectives_features = set_of_adjectives_features + adjectives
#set_of_adjectives_features = set(set_of_adjectives_features)
#f.close()
#f = open('_STEM_.txt', 'r')                                                    
# parse each review
#for line in f.readlines():
#    fs.parse(line)
#f.close()
#f = open('_HUM_.txt', 'r')
#for line in f.readlines():
#    fs.parse(line)
#f.close()
#f = open('_SOC_.txt', 'r')
#for line in f.readline():
#    fs.parse(line)
#f.close()
#fs.calc()
#set_of_adjectives_features = fs.ret_keys()
#print set_of_adjectives_features

# Modified from chapter 6 of NLTK

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

stem_featuresets = []
hum_featuresets = []
soc_featuresets = []

from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

import csv

#feature_selector = Chi_Squared()   

# read in gold standard for stem
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
        stem_featuresets.append(p)
        #feature_selector.parse(p)

# read in gold standard                                                        
with open('batch_results_hum.csv','rU') as csvfile:
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
        hum_featuresets.append(p)
        #feature_selector.parse(p)

# read in gold standard for soc                                                        
with open('batch_results_soc.csv','rU') as csvfile:
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
        soc_featuresets.append(p)
        #feature_selector.parse(p)

#feature_selector.build_count_matrix()
#features = feature_selector.features()

final_featuresets = []
#import random
#stem_featuresets = random.shuffle(stem_featuresets)
#hum_featuresets = random.shuffle(hum_featuresets)
#soc_featuresets = random.shuffle(soc_featuresets)
#final_featuresets = stem_featuresets[300:]
#final_featuresets += hum_featuresets[300:]
#final_featuresets += soc_featuresets[300:]
featuresets = hum_featuresets

for review,sentiment in featuresets:
    fs.parse(review,sentiment)

fs.calc()
print (fs.ret_keys())

final_featuresets = []

# update to only include Chi-Squared selected features
for review,sentiment in featuresets:
    final_featuresets.append((review_features(review),sentiment))

# divide featuresets
len_of_featuresets = len(final_featuresets)
three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)
train_set, devtest_set = final_featuresets[three_quarters:], final_featuresets[one_quarter:three_quarters]
test_set = []

for review,sentiment in soc_featuresets[:200]:
    test_set.append((review_features(review),sentiment))

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

# Examples of Misclassified Reviews
# TODO: DEBUG
#errors = []
#for review_features, sentiment in test_set:
 #   guess = classifier.classify(review_features)
  #  if guess != sentiment:
   #     errors.append((sentiment,guess,review))
#for (sentiment,guess,review) in sorted(errors):
 #   print sentiment
  #  print guess
   # print review