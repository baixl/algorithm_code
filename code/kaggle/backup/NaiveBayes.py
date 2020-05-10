#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xiaolingbai@gmail.com
file: NaiveBayes
date: 2017/10/11
bref: 朴素贝叶斯算法demo
"""

# 读取新闻文本数据集
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

print len(news.data)
print news.data[0]


# 对数据进行随机采样
from sklearn.model_selection import train_test_split
#随机采样25%的数据作为测试集
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

#使用朴素贝叶斯分类器对新闻文本数据进行预测
#首先进行特征转化
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

# 导入朴素贝叶斯，这里使用 多项式模型
from sklearn.naive_bayes import MultinomialNB
mnb  = MultinomialNB()
# 利用模型数据对参数进行估计
mnb.fit(x_train, y_train)

# 对测试样本进行预测没结果存储在y_pred 中
y_pred = mnb.predict(x_test)

from sklearn.metrics import classification_report
print "平均准确率", mnb.score(x_test, y_test)
print classification_report(y_test, y_pred, target_names=news.target_names)