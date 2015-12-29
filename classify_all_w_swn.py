import nltk, re, pprint
from nltk import word_tokenize

# Chi-Square Feature Selection
#import numpy as np

from nltk.corpus import sentiwordnet as swn

# modified from https://de.dariah.eu/tatom/feature_selection.html
# and https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/                                    
class Feature_Selection(object):

    def __init__(self):
        self.all_words = set([])
        self.dcount = 0
        self.s = 0
        self.n = 0
        labels = []

    def parse(self, doc):
        tokens = word_tokenize(doc)
        words = [w.lower() for w in tokens]
        pos_tagged_words = nltk.pos_tag(words)
        adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
        set_adjectives = set(adjectives)
        all_w = self.all_words | set_adjectives
        self.all_words = all_w
        self.dcount += 1

    # top 30 chi-squared values
    def ret_keys(self):
        k = self.keys
        return k # above threshold
        
    def calc(self):
        keys = []
        for key in self.all_words:
            scores = swn.senti_synsets(key, 'a')
            if not list(scores):
                continue
            score = list(scores)[0]
            objectivity = score.obj_score()
            if objectivity < 0.5:
                keys.append(key)
        self.keys = keys

fs = Feature_Selection()
f = open('_STEM_.txt', 'r')                                                    
# parse each review
for line in f.readlines():
    fs.parse(line)
f.close()
f = open('_HUM_.txt', 'r')
for line in f.readlines():
    fs.parse(line)
f.close()
f = open('_SOC_.txt', 'r')
for line in f.readline():
    fs.parse(line)
f.close()
fs.calc()
set_of_adjectives_features = fs.ret_keys()
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
    for adj in set_of_adjectives_features:
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
        p = (review_features(review),sentiment)
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
        p = (review_features(review),sentiment)
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
        p = (review_features(review),sentiment)
        soc_featuresets.append(p)
        #feature_selector.parse(p)

#feature_selector.build_count_matrix()
#features = feature_selector.features()

final_featuresets = []
#import random
#stem_featuresets = random.shuffle(stem_featuresets)
#hum_featuresets = random.shuffle(hum_featuresets)
#soc_featuresets = random.shuffle(soc_featuresets)
final_featuresets = stem_featuresets[300:]
final_featuresets += hum_featuresets[300:]
final_featuresets += soc_featuresets[300:]

# update to only include Chi-Squared selected features
#for review in featuresets:
    #review_features, _ = review
    #new_features = {}
    #for adj in features:
    #    new_features['contains({})'.format(adj)] = review_features['contains({})'.format(adj)]
    #final_featuresets.append(new_features)

# divide featuresets
len_of_featuresets = len(final_featuresets)
three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)
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