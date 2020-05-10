# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
author: xiaolingbai@gmail.com
file: titanic_prediction
date: 2017/8/24
bref:
"""

import pandas as pd
import numpy as np

# 读取训练集和测试集
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# step1：查看数据的基本信息
print '训练集:-----------------：\n', train_data.info()
print '测试集:-----------------\n', test_data.info()

# step2：特征选择
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

x_train = train_data[selected_features]
x_test = test_data[selected_features]
y_train = train_data['Survived']

print x_train['Embarked'].value_counts()
print x_test['Embarked'].value_counts()

# step3：数据预处理
# 使用出现频率最高的 S填充Embarked的缺省值
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# 使用品均值填充Age
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
print  '查看处理后数据:-----------------'
print x_train.info()
print x_test.info()

# step4：特征向量化
from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print "特征向量化结束后：:-----------------",dict_vec.feature_names_
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

# step5: 导入模型训练
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

# step6: 进行预测
# 使用5折交叉验证对rfc进行评估

from sklearn.model_selection import cross_val_score

print cross_val_score(rfc, x_train, y_train, cv=5).mean()

rfc.fit(x_train, y_train)

rfc_y_prediction = rfc.predict(x_test)
rfc_submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': rfc_y_prediction})
rfc_submission.to_csv('./gender_submission.csv', index=False)
