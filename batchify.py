# output first 200 reviews

import re, pprint

f = open('_STEM_.txt')

for x in range(0,200):
    print f.readline()
