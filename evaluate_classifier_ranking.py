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
        fs.parse(review,sentiment)

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
        fs.parse(review,sentiment)

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
        fs.parse(review,sentiment)

#feature_selector.build_count_matrix()
#features = feature_selector.features()
fs.rem_nonevaluative()
fs.calc()
final_featuresets = []

for review,sentiment in hum_featuresets:
    final_featuresets.append((review_features(review),sentiment))
for review,sentiment in stem_featuresets:
    final_featuresets.append((review_features(review),sentiment))
for review,sentiment in soc_featuresets:
    final_featuresets.append((review_features(review),sentiment))

#from random import shuffle
#shuffle(final_featuresets)
#import random
#stem_featuresets = random.shuffle(stem_featuresets)
#hum_featuresets = random.shuffle(hum_featuresets)
#soc_featuresets = random.shuffle(soc_featuresets)
#final_featuresets = stem_featuresets[300:]
#final_featuresets += hum_featuresets[300:]
#final_featuresets += soc_featuresets[300:]

# update to only include Chi-Squared selected features
#for review in featuresets:
    #review_features, _ = review
    #new_features = {}
    #for adj in features:
    #    new_features['contains({})'.format(adj)] = review_features['contains({})'.format(adj)]
    #final_featuresets.append(new_features)

#TRIAL 1
train_set = final_featuresets

# classify
classifier = nltk.NaiveBayesClassifier.train(train_set)

# compute ranking
rankings = []

f = open('random10/1.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("1",percent_satisfied))

f = open('random10/2.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("2",percent_satisfied))

f = open('random10/3.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("3",percent_satisfied))

f = open('random10/4.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("4",percent_satisfied))

f = open('random10/5.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("5",percent_satisfied))

f = open('random10/6.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("6",percent_satisfied))

f = open('random10/7.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("7",percent_satisfied))

f = open('random10/8.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("8",percent_satisfied))

f = open('random10/9.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("9",percent_satisfied))

f = open('random10/10.txt', 'r')                                                    
# parse each review
count = 0
sat = 0
for line in f.readlines():
    sent = classifier.classify(review_features(line))
    if sent == "Satisfied":
        sat += 1
    count += 1
f.close()
percent_satisfied = float(sat)/count
rankings.append(("10",percent_satisfied))

rank = sorted(rankings, key=lambda score: score[1], reverse=True)

print rank