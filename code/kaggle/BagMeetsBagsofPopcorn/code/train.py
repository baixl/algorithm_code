#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: train
date: 2018/3/30
bref:逻辑斯蒂回归+tf-idf
"""
import pandas as pd
import time
import numpy as np

start_time = time.time()
train = pd.read_csv('../input/clean_labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../input/clean_testData.tsv', delimiter='\t')

# 对train 和test分别进行上述处理
x_train = []
for review in train['review']:
    x_train.append(review)
x_test = []
for review in test['review']:
    x_test.append(review)
y_train = train['sentiment']

print("数据加载完毕, 时间")

# 使用Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2,  # 最小支持度
                        max_features=None,
                        strip_accents='unicode',
                        analyzer='word',
                        token_pattern=r'\w{1,}',
                        ngram_range=(1, 3),  # 二元文法模型
                        use_idf=1,
                        smooth_idf=1,
                        sublinear_tf=1,
                        stop_words='english')
data_all = x_train + x_test
len_train = len(x_train)

tfidf.fit(data_all)  # 拟合
data_all = tfidf.transform(data_all)  # 转化
# 回复成训练集和测试集
x_train = data_all[:len_train]
x_test = data_all[len_train:]

print("tf-idf 处理结束")

# 使用逻辑斯谛回归进行训练
start_time = time.time()
from sklearn.linear_model import LogisticRegression  as LR
from sklearn.grid_search import GridSearchCV

# 设定grid search的参数
grid_values = {'C': [30]}
# model_LR = LogisticRegression(LR(penalty='L2', dual=True, random_state=0)

# from sklearn.cross_validation import cross_val_score
# print("多项式贝叶斯10折交叉得分",
#       np.mean(cross_val_score(model_NB, x_train, y_train, cv=10, scoring='roc_auc')))


# 设定打分为roc_auc
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
model_LR.fit(x_train, y_train)
# 20折交叉验证
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True,
                                 fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001),
             fit_params={}, iid=True, n_jobs=1,
             param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
             scoring='roc_auc', verbose=0)
# 输出结果
print(model_LR.grid_scores_)

test_predicted = np.array(model_LR.predict(x_test))
print('保存结果...')
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv('lr_output.csv', index=False)
print('结束.')
