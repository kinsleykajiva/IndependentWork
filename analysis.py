# determine average length of review

import pprint

f = open("_STEM_.txt")

raw = f.readline()
count = 0

while raw != "":
     if len(raw) > 1:
          count += 1
     raw = f.readline()

print count

f.close()

f = open("_SOC_.txt")

raw = f.readline()
count = 0

while raw != "":
     if len(raw) > 1:
          count += 1
     raw = f.readline()

print count

f.close()

f = open("_HUM_.txt")

raw = f.readline()
count = 0

while raw != "":
     if len(raw) > 1:
          count += 1
     raw = f.readline()

print count

f.close()
