import nltk, re, pprint
from nltk import word_tokenize
document = '_HUM_.txt'

from numpy import zeros
from numpy.linalg import svd
from math import log
from numpy import asarray, sum
from numpy import matrix

# perform LSA with TF-IDF weighting
# modified from https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/                                    
class LSA(object):

    def __init__(self):
        self.wdict = {}
        self.dcount = 0

    # bag of adjectives representation 
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


    def build_count_matrix(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1

    # TODO: Debug       
    def TFDIF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        #print WordsPerDoc
        #print DocsPerWord
        rows, cols = self.A.shape
        #print rows
        #print cols
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = self.A[i,j] #(float(self.A[i,j]) / WordsPerDoc[j]) * log(float(cols)/DocsPerWord[i])
                                            
    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)
    
    # TODO: DEBUG
    def term_similarity_matrix(self):
        top_singular_values = [i for i in self.S if i > 1]
        length = len(top_singular_values) - 1
        new_matrix = zeros([length,length])
        for i in xrange(0,length):
            for j in xrange(0, length):
         return (self.U).dot(new_matrix)

    # TODO: TRUE IMPLEMENTATION
    def top25(self):
        l = zip(self.keys,self.S)
        return [w for w,f in sorted(l, key=lambda word: word[1], reverse=True)][1:26]

    def printA(self):
        print self.A[0:10, 0]
        print self.A[0, 0:10]
        print self.A

    def printSVD(self):
        print 'Here are the singular values'
        print self.S
        print 'Here are the first 3 columns of the U matrix'
        print -1*self.U[:, 0:3]
        print 'Here are the first 3 rows of the Vt matrix'
        print -1*self.Vt[0:3, :]

lsa = LSA() 
f = open(document, 'r')                                                    
# parse each review
for line in f.readlines():
    lsa.parse(line)
f.close()
lsa.build_count_matrix()
lsa.printA()
lsa.TFDIF()
lsa.calc()
print lsa.term_similarity_matrix()                                                       
# take top 25 features
set_of_adjectives_features = lsa.top25()

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
        featuresets.append((review_features(review),sentiment))

# divide featuresets
len_of_featuresets = len(featuresets)
three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)
train_set, test_set, devtest_set = featuresets[three_quarters:], featuresets[:one_quarter], featuresets[one_quarter:three_quarters]

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
for review_features, sentiment in test_set:
    guess = classifier.classify(review_features)
    if guess != sentiment:
        errors.append((sentiment,guess,review))
for (sentiment,guess,review) in sorted(errors):
    print sentiment
    print guess
    print review
