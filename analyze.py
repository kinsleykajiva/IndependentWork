import nltk, re, pprint
from nltk import word_tokenize
f = open('HUMstd.txt')
raw = f.read()
tokens = word_tokenize(raw)
words = [w.lower() for w in tokens]
from nltk.stem.porter import *
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in words]
fd = nltk.FreqDist(stemmed)
types = set(words)
typesstemmed = set(stemmed)
print len(types)
print len(typesstemmed)
print sorted(fd.items(), key=lambda word: word[1], reverse=True)[0:60] 
#pos_tagged_words = nltk.pos_tag(tokens)
#propernounless = []
#for word,pos in pos_tagged_words:
 #   if pos != "NNP":
  #      print word + " "
   # else:
    #    print "NAME "
#propernounless = [word for word,pos in pos_tagged_words if pos != "NNP"]
#print ' '.join(propernounless)
