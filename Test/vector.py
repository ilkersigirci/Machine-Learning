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
vocabulary=[]
tokenized=[]
words=[]

mindf=0.25
maxdf=0.75

lengthContents=len(contents)

for i in range(lengthContents):    
    tokenized.append(nltk.word_tokenize(contents[i]))
    
for i in range(lengthContents):
    count=0
    for j in range(len(tokenized[i])):
        words.append(tokenized[i][j])

lengthWords=len(words)
for i in range(lengthWords):
    count=0
    for j in range(lengthWords):
        if(words[i]==words[j]): count=count+1
    
    count=count/float(lengthContents)
    if(count>=mindf and count<=maxdf):
        vocabulary.append(words[i])

print vocabulary
