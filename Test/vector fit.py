import nltk
import numpy as np
import math
from collections import Counter

contents =[
"this is the first document",
"this document is the second document",
"and this is the third one",
"is this the first document",
]
# and third one second document first
vocabulary=[]
tokenized=[]
c=Counter()
min_word_length=3
mindf=0.25
maxdf=0.75
size=len(contents)
"""

lengthWords=len(words)
for i in range(lengthWords):
    count=0
    for j in range(lengthWords):
        if(words[i]==words[j]): count=count+1
    
    count=count/float(lengthContents)
    if(count>=mindf and count<=maxdf):
        vocabulary.append(words[i]) """

""" counted = Counter(map(tuple,tokenized))
multituples = [tuple(l) for l in tokenized]
print counted.get('document', 'not found!') """

for i in contents:    #tokenized
    tokenized.append(nltk.word_tokenize(i))

for i in range(size):   #remove duplicates
    tokenized[i]=list(set(tokenized[i]))

for i in tokenized:
    c.update(i)

for i in c:
    count=c[i]/float(size)
    length=len(i)
    if(count>=mindf and count<=maxdf and length>=min_word_length):
        vocabulary.append(i)
    

print vocabulary