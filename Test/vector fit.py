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
setTokenized=[]
term_df_dict={}
c=Counter()
min_word_length=3
mindf=0
maxdf=1
size=len(contents)

""" counted = Counter(map(tuple,tokenized))
multituples = [tuple(l) for l in tokenized]
print counted.get('document', 'not found!') """

for i in contents:    #tokenized
    tokenized.append(nltk.word_tokenize(i))

for i in range(size):   #remove duplicates
    setTokenized.append(list(set(tokenized[i])))

for i in setTokenized:
    c.update(i)

for i in c:
    count=c[i]/float(size)
    length=len(i)
    if(count>=mindf and count<=maxdf and length>=min_word_length):
        vocabulary.append(i)

for i in c:
    if i in vocabulary:
        term_df_dict[i]=c[i]/float(size)

""" print vocabulary
print term_df_dict """
