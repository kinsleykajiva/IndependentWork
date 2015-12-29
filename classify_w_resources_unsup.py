import nltk, re, pprint
import math
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *
from copy import deepcopy

# based on the methods described in "Extracting Resource Terms for Sentiment Analysis"
# http://www.aclweb.org/anthology/I11-1131
# feature based aggregation, unsupervised or supervised method

# build list of quantifiers
quantifiers = ["some","several","numerous","many","much","more","most","less","least","a large number of","a huge number of","a small number of","a tiny number of","a large amount of","a huge amount of","a small amount of","a tiny amount of","lot of","lots of","tons of","ton of","plenty of","deal of","load of","loads of","few","little"]

# in unsupervised version, if closer (by wordnet path) to less, positive; more, negative

# list of seed stemmed domain resource terms
resources = ["assignment","office hours","paper","test","exam","reading","quiz","material"]

# use WordNet to find synonyms to improve results

from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

import csv

triples = {}
for q in quantifiers:
	for r in resources:
		triples['contains({},{})'.format(q,r)] = False

featuresets = []

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

        sentences = review.split('.')
        #print sentences
        features = deepcopy(triples)
        for sent in sentences:
            tokens = word_tokenize(sent)
            words = [w.lower() for w in tokens]
            stemmer = PorterStemmer()
            stemmed = [stemmer.stem(word) for word in words]
			#pos_tagged_words = nltk.pos_tag(stemmed)

            for i,w in enumerate(stemmed):
                if w in resources and i > 0 and stemmed[i] in quantifiers:
                    features['contains({},{})'.format(stemmed[i],w)] = True

        featuresets.append((features,sentiment))

len_of_featuresets = len(featuresets)
#print len_of_featuresets
#three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)
train_set = featuresets[(one_quarter*3):]
test_set = featuresets[(3*one_quarter):len_of_featuresets]

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
