import nltk, re, pprint
from nltk import word_tokenize
document = '_HUM_.txt'

from numpy import zeros
from numpy.linalg import svd
from math import log
from numpy import asarray, sum
# perform LSA, singular value decomposition                                     
class LSA(object):
    def __init__(self):
        self.wdict = {}
        self.dcount = 0
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
                
    def TFDIF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols)/DocsPerWord[i])
                                            
    def calc(self):
        # U is the words dimensions, Vt documents, S how many concepts include
        self.U, self.S, self.Vt = svd(self.A)
    
    # dimensionality reduction - should retain 90% energy of it so,
    # 90% sum of squares
    def top25(self):
        # first 25 rows of U's words 
        # throw out first dimension
        l = zip(self.keys,self.S)
        return [w for w,f in sorted(l, key=lambda word: word[1], reverse=True)][1:26]
        
    def keyprint25(self):
        return self.keys[0:25]

    def printSVD(self):
        print 'Here are the singular values'
        print self.S
        print 'Here are the first 3 columns of the U matrix'
        print -1*self.U[:, 0:3]
        print 'Here are the first 3 rows of the Vt matrix'
        print -1*self.Vt[0:3, :]

lsa = LSA() 
# build term-document matrix
f = open(document, 'r')                                                    
# parse each review
for line in f.readlines():
    lsa.parse(line)
f.close()
lsa.build_count_matrix()
lsa.calc()
#lsa.printSVD()                                                       
# take top 25 
set_of_adjectives_features = lsa.top25()

# unstemmed
# feature extraction - adjectives - UPDATE FOR EVALUATIVE ADJECTIVES USING CONTEXT
def review_features(review):
    tokens = word_tokenize(review)
    words = [w.lower() for w in tokens]
    # choice of pos tagger
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
with open('batch_results_hum.csv','rU') as csvfile:
    # read in review, sentiment agreed upon from excel file
    reader = csv.DictReader(csvfile)
    # change to Satisfied or not so can include all data
    for row1,row2,row3 in grouper(reader, 3):
        review = row1['Review']
        sentiment1 = row1['Answer']
        sentiment2 = row2['Answer']
        sentiment3 = row3['Answer']
        sentiment = "Neither Satisfied Nor Dissatisfied"
        # test for agreement
        if (sentiment1 == sentiment2):
            sentiment = sentiment1
        elif (sentiment2 == sentiment3):
            sentiment = sentiment2
        elif (sentiment1 == sentiment3):
            sentiment = sentiment3
        #else:
         #   print sentiment1
          #  print sentiment2
           # print sentiment3
            #print sentiment
        # exclude neutral reviews, building binary classifer
        if sentiment == "Neither Satisfied Nor Dissatisfied":
            continue
        else:
            featuresets.append((review_features(review),sentiment))

# error analysis
#import random
#random.shuffle(featuresets)
len_of_featuresets = len(featuresets)
three_quarters = int(.75*len_of_featuresets)
one_quarter = int(.25*len_of_featuresets)
train_set, test_set, devtest_set = featuresets[three_quarters:], featuresets[:one_quarter], featuresets[one_quarter:three_quarters]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# calculate f-measure
# print (nltk.classify.accuracy(classifier,test_set))
tp = 0
fp = 0
fn = 0

for review_features,sentiment in test_set:
    if sentiment == "Satisfied":
        if classifier.classify(review_features) == "Satisfied":
            tp += 1
        else:
            fn += 1
    else:
        if classifier.classify(review_features) == "Satisfied":
            fp +=1


print tp
print fp
print fn
#precision = float(tp/(tp+fp))
#recall = float((tp)/(tp+fn))
#print precision
#print recall
#fmeasure = 2.0*float(((precision*recall)/(precision+recall)))
#print fmeasure

# errors
errors = []
#print devtest_set
for review_features, sentiment in devtest_set:
    guess = classifier.classify(review_features)
    if guess != sentiment:
        errors.append((sentiment,guess,review))
for (sentiment,guess,review) in sorted(errors):
    print sentiment
    print guess
    print review
