# -*- coding:utf-8 -*-
# !/usr/bin/env python


"""
author: xiaolingbai@gmail.com
file: svmdemo
date: 2017/8/20
bref: 利用svm进行手写数字识别，多分类
"""

# 从sklearn加载手写数字数据
from sklearn.datasets import load_digits

digits = load_digits()
print digits.data.shape

# 将数据分割成训练集和测试集
from sklearn.cross_validation import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

print y_train.shape
print y_test.shape

# 使用支持向量机对手写数字进行图像识别
# 数据标准化
from sklearn.preprocessing import StandardScaler

# 导入基于线性假设的支持向量机分类器LinersSVC
from sklearn.svm import LinearSVC

ss = StandardScaler()
# 数据标准化
x_train=ss.fit_transform(x_train)
x_test =ss.transform(x_test)

lsvc = LinearSVC()

lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

print 'Accuracy of Liner SVC:',lsvc.score(x_test,y_test)

from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))