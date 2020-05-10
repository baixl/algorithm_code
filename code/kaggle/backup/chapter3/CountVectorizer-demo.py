# -*- coding:utf-8 -*-
# !/usr/bin/env python


"""
author: xiaolingbai@gmail.com
file: CountVectorizer-demo
date: 2017/8/22
bref:
"""

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size= 0.25, random_state=33)

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()

x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)


from sklearn.naive_bayes import MultinomialNB

mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)
