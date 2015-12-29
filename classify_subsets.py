import nltk, re, pprint
from nltk import word_tokenize

# Chi-Square Feature Selection
#import numpy as np

# modified from https://de.dariah.eu/tatom/feature_selection.html
# and https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/                                    
class Chi_Squared(object):

    def __init__(self):
        self.wdict = {}
        self.dcount = 0
        labels = []

    def parse(self, doc):
        tokens = word_tokenize(doc)
        words = [w.lower() for w in tokens]
        pos_tagged_words = nltk.pos_tag(words)
        adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
        set_adjectives = set(adjectives)
        for adj in set_adjectives:
            if adj in self.wdict:
                self.wdict[adj].append(self.dcount)
            else:
                self.wdict[adj] = [self.dcount]
        self.dcount += 1

    # adjectives that appear in at least 2 reviews
    def keys(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        return self.keys

    # bag of adjectives representation 
    #def parse(self, (features,sentiment)):
     #   for adj in features:
    #        labels.append(sentiment)
     #       if adj in self.wdict:
      #          self.wdict[adj].append(self.dcount) # update the word entry
      #      else:
      #          self.wdict[adj] = [self.dcount] # create a new word entry
       # self.dcount += 1 # increase the review identifier

    #def build_count_matrix(self):
       # self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
       # self.A = zeros([len(self.keys), self.dcount])
       # for i, k in enumerate(self.keys):
        #    for d in self.wdict[k]:
        #        self.A[i,d] += 1

   # def features(self):
      #  keyness, _ = chi2(self.A, self.labels)
       # ranking = np.argsort(keyness)[::-1]
       # return self.keys[ranking][0:15]

fs = Chi_Squared()
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
set_of_adjectives_features = sorted(fs.keys())
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
#final_featuresets = stem_featuresets[300:]
#final_featuresets += hum_featuresets[300:]
#final_featuresets += soc_featuresets[300:]
final_featuresets = hum_featuresets

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
train_set, devtest_set = final_featuresets[three_quarters:], final_featuresets[one_quarter:three_quarters]
test_set = soc_featuresets[:200]

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
