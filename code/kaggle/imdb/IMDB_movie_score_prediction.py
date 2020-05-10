# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
author: xiaolingbai@gmail.com 
file: IMDB_movie_score_prediction
date: 2017/8/28
bref:
"""

import pandas as pd
# 读取imdb电影数据 https://www.kaggle.com/c/word2vec-nlp-tutorial
train = pd.read_csv('./labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('./testData.tsv', delimiter='\t')
print train.head()
print test.head()
#
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
#
def review_to_text(review, remove_stop_words):
    # 去掉费html标记
    raw_text = BeautifulSoup(review, 'html').get_text()
    # 去掉非字母字符
    letters = re.sub('[^a-zA-Z]',  '' , raw_text)
    words = letters.lower().split()
    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stopwords]
    return words
#
# x_train =[]
# for review in train['review']:
#     x_train.append(' '.join(review_to_text(review,True)))
#
# x_test =[]
# for review in test['review']:
#     x_test.append(' '.join(review_to_text(review,True)))
# y_train = train['sentiment']
# # 导入文本特性抽取器
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# # 导入朴素贝叶斯模型
# from sklearn.naive_bayes import  MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.grid_search import GridSearchCV
#
#


unlabeled_train = pd.read_csv('./unlabeledTrainData.tsv', delimiter='\t', quoting=3)

corpora =[]
for review in unlabeled_train['review']:
    corpora += review_to_text(review, True)

num_features =300
min_word_count =20
num_works =4
context =10
downsampling =1e-3
