#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: sk_and_XGBoost
date: 2018/1/3
bref: 随机森林和XGBoost的demo
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

# print x_train['Embarked'].value_counts()
# print x_test['Embarked'].value_counts()

# step3：数据预处理
# 使用出现频率最高的 S填充Embarked的缺省值
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# 使用品均值填充Age
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
# print  '查看处理后数据:-----------------'
# print x_train.info()
print x_test.info()

# step4：特征向量化


dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print "特征向量化结束后：:-----------------", dict_vec.feature_names_
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

# step5: 导入模型训练


rfc = RandomForestClassifier()

# step6: 进行预测
# 使用5折交叉验证对rfc进行评估


print "随机森林准确性：", cross_val_score(rfc, x_train, y_train, cv=5).mean()

rfc.fit(x_train, y_train)
#
# rfc_y_prediction = rfc.predict(x_test)
# rfc_submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': rfc_y_prediction})
# rfc_submission.to_csv('./gender_submission.csv', index=False)

# -------------------------分割线------------------------------


# 使用默认配置
xgbc = XGBClassifier()
print "XGBoost准确性：", cross_val_score(xgbc, x_train, y_train, cv=5).mean()
print xgbc

# 使用并行网格搜索方式获取更好的超参数
#
# parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
#               'objective': ['binary:logistic'],
#               'learning_rate': [0.05],  # so called `eta` value
#               'max_depth': [6],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               'n_estimators': [5],  # number of trees, change it to 1000 for better results
#               'missing': [-999],
#               'seed': [1337]}
#
# params = {'max_depth': range(2, 3), 'n_estimators': range(50, 100, 50),
#           'learning_rate': [0.05, 0.1]}
# xgbc_best = XGBClassifier()
# gs = GridSearchCV(xgbc_best, params, n_jobs=30, cv=5, verbose=1)
# gs = GridSearchCV(xgbc_best, parameters, n_jobs=5,
#                   cv=5, scoring='roc_auc', verbose=2, refit=True)

clf = XGBClassifier(learning_rate=0.1, max_depth=2, silent=True, objective='binary:logistic')
param_test = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1),
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_test, scoring='accuracy', cv=5)

grid_search.fit(x_train, y_train)
# print grid_search.grid_scores_
print grid_search.best_params_
print grid_search.best_score_
