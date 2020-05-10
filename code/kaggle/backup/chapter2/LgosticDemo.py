# -*- coding:utf-8 -*-
# !/usr/bin/env python

"""
author: xiaolingbai@gmail.com
file: LgosticDemo
date: 2017/8/20
bref: 使用逻辑斯谛回归 和随机梯度法两种方式，进行癌症良性和恶性的预测
"""
import pandas as pd
import numpy as np

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape'
                ,'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chormatin','Normal Nucleoli',
                'Mitoses', 'Class']
# 从URL读取数据,并指定表头
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                   'breast-cancer-wisconsin/breast-cancer-wisconsin.input', names=column_names)
print data.shape
# 预处理，将?替换成缺失值
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据(只要有一个维度有缺失值，就丢弃)
data = data.dropna(how='any')

print data.shape


# 将原始数据分成训练集和测试集，通常75%作为训练，25%作为测试

# 使用sklearn.cross_valiation中的train_test_spilt用于分割数据
from sklearn.cross_validation import train_test_split
# 随机采样
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]],
            data[column_names[10]], test_size=0.25, random_state=33)
# 检查训练样本的数量和类别分布
print y_train.value_counts()
print y_test.value_counts()

# 开始训练，分别使用逻辑斯谛回归和随机梯度参数估计两种方法
#从sklearn.preprocessing 里导入StandScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier # 随机梯度下降

# 标准化数据， 使每个维度的数据 方差为1 ，均值为0， 避免某些维度数据过大，对特征值进行主导
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test =ss.transform(x_test)

# 初始化LogisticRegression 与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
# 调用LogisticRegression中的fit函数 训练模型参数
lr.fit(x_train,y_train)
# 使用训练好的模型lr对x_test进行预测，结果存储在lr_y_predict中
lr_y_predict = lr.predict(x_test)

sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)


# 性能分析，指标，准确率召回率F值等

from sklearn.metrics import classification_report

# 使用逻辑斯谛回归模型自带的评分函数score获得模型在测试集上的准确性结果
print 'Accuracy of Logstic:', lr.score(x_test, y_test)
# 利用 classification_report 获得逻辑斯谛模型的其他指标
print classification_report(y_test, lr_y_predict, target_names=['良性2', '恶性4'])

# 随机梯度下降模型的评估
# 使用随机梯度下降模型自带的评分函数score 评价准确性
print 'Accuracy of SGD:', sgdc.score(x_test,y_test)
# 利用 classification_report 获得SGDC的其他三个指标
print classification_report(y_test,sgdc_y_predict, target_names=['良性2', '恶性4'])