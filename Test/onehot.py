import numpy as np


labels=["Action","Comedy","Action","Crime","Comedy"]
tags=[]

for i in labels:
    result=0
    for j in tags:
        if(i==j):
            result=1
            break
    if(result): continue
    tags.append(i)

#print labels
print tags

#transform

sLabels=len(labels)
sTags=len(tags)
result=np.zeros((sLabels,sTags),dtype=int)

for i in range(sLabels):
    for j in range(sTags):
        if(labels[i]==tags[j]):
            result[i][j]=1

#print result

#decode

array=[0,1,0]
for i in range(len(array)):
    if(array[i]):
         print tags[i]