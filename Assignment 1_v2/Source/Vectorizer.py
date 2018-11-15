#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy as np
import math
from collections import Counter

class Vectorizer:
    def __init__(self, min_word_length=3, max_df=1.0, min_df=0.0):
        self.min_word_length = min_word_length
        self.max_df=max_df
        self.min_df=min_df
        self.term_df_dict = {}

    def fit(self, raw_documents):
        """Generates vocabulary for feature extraction. Ignores words shorter than min_word_length and document frequency
        not between max_df and min_df.

        :param raw_documents: list of string for creating vocabulary
        :return: None
        """
        self.document_count = len(raw_documents)
        # TODO: Implement this method
        tokenized=[]
        c=Counter()
        size=len(raw_documents)
        self.vocabulary=[]
        for i in raw_documents:    #tokenized
            tokenized.append(nltk.word_tokenize(i))
        for i in range(size):   #remove duplicates
            tokenized[i]=list(set(tokenized[i]))
        for i in tokenized: #count
            c.update(i)
        for i in c:
            count=c[i]/float(size)
            length=len(i)
            if(count>=self.min_df and count<=self.max_df and length>=self.min_word_length):
                self.vocabulary.append(i)
        
        return self.vocabulary

    def _transform(self, raw_document, method):
        """Creates a feature vector for given raw_document according to vocabulary.

        :param raw_document: string
        :param method: one of count, existance, tf-idf
        :return: numpy array as feature vector
        """
        # TODO: Implement this method
        pass

    def transform(self, raw_documents, method="tf-idf"):
        """For each document in raw_documents calls _transform and returns array of arrays.

        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        # TODO: Implement this method
        pass

    def fit_transform(self, raw_documents, method="tf-idf"):
        """Calls fit and transform methods respectively.

        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        # TODO: Implement this method
        pass

    def get_feature_names(self):
        """Returns vocabulary.

        :return: list of string
        """
        try:
            self.vocabulary
        except AttributeError:
            print "Please first fit the model."
            return []
        return self.vocabulary

    def get_term_dfs(self):
        """Returns number of occurances for each term in the vocabulary in sorted.

        :return: array of tuples
        """
        return sorted(self.term_df_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)

if __name__=="__main__":
    v = Vectorizer(min_df=0, max_df=1)
    contents = [
     "this is the first document",
     "this document is the second document",
     "and this is the third one",
     "is this the first document",
 ]
    v.fit(contents)
    print v.get_feature_names()
    existance_vector = v.transform(contents, method="existance")        
    print existance_vector
    count_vector = v.transform(contents, method="count")        
    print count_vector
    tf_idf_vector = v.transform(contents, method="tf-idf")
    print tf_idf_vector