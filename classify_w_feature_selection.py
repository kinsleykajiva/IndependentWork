import nltk, re, pprint
import math
from nltk import word_tokenize

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
        #print len(self.keys)

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
# Modified from chapter 6 of NLTK

f = open('_SOC_.txt', 'r')
adj_features = []
# parse each review                                                            
for line in f.readlines():
    tokens = word_tokenize(line)
    words = [w.lower() for w in tokens]                                        
    pos_tagged_words = nltk.pos_tag(words)
    adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
    set_adjectives = set(adjectives)
    adj_features += set_adjectives
set_of_adjectives_features = set(adj_features)

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

featuresets = []

from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

import csv

#feature_selector = Chi_Squared()   

# read in gold standard
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
        featuresets.append(p)
        fs.parse(review, sentiment)

#feature_selector.build_count_matrix()
#features = feature_selector.features()

#print featuresets[0:1]

#final_featuresets = []
fs.calc()
my_keys = fs.ret_keys()

#print len(my_keys)

final_featuresets = []

# update to only include Chi-Squared selected features
for review in featuresets:
    review_features, sentiment = review
    new_features = {}
    for adj in my_keys:
        if 'contains({})'.format(adj) in review_features:
            new_features['contains({})'.format(adj)] = review_features['contains({})'.format(adj)]
            #print review_features['contains({})'.format(adj)]
        else:
            new_features['contains({})'.format(adj)] = False
    final_featuresets.append((new_features,sentiment))
    #print count

#print final_featuresets[0:1]

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

# Examples of Misclassified Reviews
# TODO: DEBUG
errors = []
#for review_features, sentiment in test_set:
 #   guess = classifier.classify(review_features)
  #  if guess != sentiment:
   #     errors.append((sentiment,guess,review))
#for (sentiment,guess,review) in sorted(errors):
 #   print sentiment
  #  print guess
   # print review
