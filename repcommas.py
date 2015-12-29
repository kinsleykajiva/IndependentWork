# replace commas with dashes

import re, pprint

f = open("_STEM_.txt")

raw = f.read()

s = raw.replace(",", "-")

s = re.sub("[\n]+", ",", s)

print s
