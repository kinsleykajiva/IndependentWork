import nltk, re, pprint
import math
from nltk import word_tokenize

file = "allSOC.txt"
lines = 0
words = 0

# determine average length of review
with open(file) as f:
    for line in f:
        words += len(word_tokenize(line))
        lines+=1

avg = words/lines

sumofsqrs = 0

# calculate std deviation
with open(file) as f:
    for line in f:
        w = len(word_tokenize(line))
        sumofsqrs += (abs(w-avg))*(abs(w-avg))

variance = sumofsqrs / lines

stddev = math.sqrt(variance)
print stddev
max = avg + stddev
min = avg - stddev

lines = 0

#with open(file) as f:
 #   for line in f:
  #      w = len(word_tokenize(line))
   #     if w >= min and w <= max:
    #        lines += 1

#print lines

# include only those reviews within one std deviation 
#with open(file) as f:
 #   for line in f:
  #      w = len(word_tokenize(line))
   #     if w >= min and w <= max:
    #        print line
