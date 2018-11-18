import numpy as np
import nltk
contents =[
"this is the first document",
"this document is the second document",
"and this is the third one",
"is this the first document",
]
vocabulary=['and', 'third', 'one', 'second', 'document', 'first']
""" size=len(contents)
tokenized=[]
setTokenized=[]
for i in contents:    #tokenized
    tokenized.append(nltk.word_tokenize(i))
for i in range(size):   #remove duplicates
    setTokenized.append(list(set(tokenized[i])))
row=len(contents)
column=len(vocabulary)  #self.vocabulary 
vector=np.zeros((row,column),dtype=int)



                
print result



        temp = nltk.tokenize.word_tokenize(raw_document)
        result = np.zeros((len(self.vocabulary)))
        if method == "existance":
            for word in temp:
                if word in self.vocabulary:
                    result[self.vocabulary.index(word)] = 1
        elif method == "count":
            for word in temp:
                if word in self.vocabulary:
                    result[self.vocabulary.index(word)] +=1
        elif method == "tf-idf":
            for word in temp:
                if word not in self.vocabulary:
                    continue
                freq = self.term_df_dict[word]*self.document_count
                idf = math.log((1+self.document_count)/(1+freq)) + 1
                tf = 0
                for i in range(len(temp)):
                    if temp[i] == word:
                        tf+=1
                result[self.vocabulary.index(word)] = tf*idf
            n = np.linalg.norm(result)
            for i in range(len(result)):
                result[i] = result[i]/n    
        return result
 """


a="this is forgive the give"
print a.count("give")

temp = nltk.tokenize.word_tokenize(a)
print temp.count("give")