import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string


data="can't"
stopWords=set(stopwords.words('english'))
tokenized=word_tokenize(data)
noPunc=[]
noStop=[]

for i in tokenized:
    new_i=i.translate(None, string.punctuation)
    noPunc.append(new_i)

for i in noPunc:
    if i not in stopWords:
        noStop.append(i)


#remove Number

