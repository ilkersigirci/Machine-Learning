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
words=[]
min_word_length=3
mindf=0.25
maxdf=0.75
"""

lengthWords=len(words)
for i in range(lengthWords):
    count=0
    for j in range(lengthWords):
        if(words[i]==words[j]): count=count+1
    
    count=count/float(lengthContents)
    if(count>=mindf and count<=maxdf):
        vocabulary.append(words[i]) """

""" lengthContents=len(contents)
for i in range(lengthContents):    
    tokenized.append(nltk.word_tokenize(contents[i]))
print tokenized
print sum(x.count('document') for x in tokenized) """

""" counted = Counter(map(tuple,tokenized))
multituples = [tuple(l) for l in tokenized]
print counted.get('document', 'not found!') """

lengthContents=len(contents)
for i in range(lengthContents):    
    tokenized.append(nltk.word_tokenize(contents[i]))

for i in range(lengthContents):
        for j in range(len(tokenized[i])):
                word=tokenized[i][j]
                if(len(word)<min_word_length):continue
                count=1
                for k in tokenized:
                        if word in k:
                                if(k==lengthContents):break
                                if(k==i):continue
                                if(j==word):count=count+1
                                k=k+1
                count=count/float(lengthContents)
                if(count>=mindf and count<=maxdf):
                        vocabulary.append(words[i])

print vocabulary