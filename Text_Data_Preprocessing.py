# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:32:48 2021

@author: vishal
"""

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer


def get_lemmatizer():
    lmtzr = WordNetLemmatizer()
    return lmtzr

def data_preprocessing(df,text_col):
    text_list = []
    
    lemmatizer = get_lemmatizer()
    
    for i in range(len(df)):
        sent1 = re.sub('[^a-zA-Z]',' ',df[text_col][i])
        word_tokens = word_tokenize(sent1.lower())
        
        tokens = [lemmatizer.lemmatize(word) for word in word_tokens if word not in set(stopwords.words('english'))]
        
        tokens = ' '.join(tokens)
        text_list.append(tokens)
        
    return text_list

def get_countvectorizer():
    return CountVectorizer()

def BOW(text_list):
    count_vec = get_countvectorizer()
    features = count_vec.fit_transform(text_list).toarray()
    return features


