# uses PyChant spellchecking library for Python
# checks if word is both capitalized and not in US English dictionary or a NNP
# if so, replaces with a "___"

import enchant
d = enchant.Dict("en_US")
import nltk, re, pprint
from nltk import word_tokenize
f = open('HUMstd.txt')

raw = f.readline()

# while there is a review to be read
while raw != "":
    tokens = word_tokenize(raw)
    pos_tagged_words = nltk.pos_tag(tokens)                                   
    
#((not d.check(word)) and word[0].isupper())#
#or (index > 0 and pos_tagged_words[index - 1][1][0] == "Professor"))#
#check if possessive?#                                                  
    for word,pos in pos_tagged_words:

        if pos == "NNP" and (not d.check(word) and word[0].isupper()):       
            print "___",
        else:                                                                             print word + " ",
        
    print ''
    raw = f.readline()
    

f.close()                                                          
#propernounless = [word for word,pos in pos_tagged_words if pos != "NNP"]      #print ' '.join(propernounless)  
            
