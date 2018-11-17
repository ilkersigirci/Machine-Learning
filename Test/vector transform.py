import numpy as np
import nltk
contents =[
"this is the first document",
"this document is the second document",
"and this is the third one",
"is this the first document",
]
vocabulary=['and', 'third', 'one', 'second', 'document', 'first']
size=len(contents)
tokenized=[]
setTokenized=[]
for i in contents:    #tokenized
    tokenized.append(nltk.word_tokenize(i))
for i in range(size):   #remove duplicates
    setTokenized.append(list(set(tokenized[i])))
row=len(contents)
column=len(vocabulary)  #self.vocabulary 
result=np.zeros((row,column),dtype=int)

""" for i in range(row):
    for j in range(column):
        if(tokenized[i]==vocabulary[j]):
            result[i][j]=1 """

for i in tokenized:
    for j in i:
        print j


                
print result


