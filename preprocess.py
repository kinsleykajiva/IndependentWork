import nltk, re, pprint
from nltk import word_tokenize
f = open('_HUM_.txt')
raw = f.read()
tokens = word_tokenize(raw)
words = [w.lower() for w in tokens]
#from nltk.stem.porter import *
#stemmer = PorterStemmer()
#stemmed = [stemmer.stem(word) for word in words]
#fd = nltk.FreqDist(stemmed)
#types = set(words)
#typesstemmed = set(stemmed)
#print len(types)
#print len(typesstemmed)
#pos_tagged_words = nltk.pos_tag(words)
#adjectives = [w for w,p in pos_tagged_words if p == "JJ"]
#words = [w.lower() for w in adjectives]
fd = nltk.FreqDist(words)
l = sorted(fd.items(), key=lambda word: word[1], reverse=True) 
for word, freq in l:
    if word == "better" or word == "best" or word == "good":
        print word,
        print '\t',
        print freq
#types = 0
#count = 0
#for word, freq in l:
#    count = count + freq
#    types = types + 1
#print count
#print types

#pos_tagged_words = nltk.pos_tag(tokens)
#propernounless = []
#for word,pos in pos_tagged_words:
 #   if pos != "NNP":
  #      print word + " "
   # else:
    #    print "NAME "
#propernounless = [word for word,pos in pos_tagged_words if pos != "NNP"]
#print ' '.join(propernounless)
