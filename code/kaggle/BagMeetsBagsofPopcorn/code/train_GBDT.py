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
                        sublinear_tf=True,
                        stop_words='english')
data_all = x_train + x_test
len_train = len(x_train)

tfidf.fit(data_all)  # 拟合
data_all = tfidf.transform(data_all)  # 转化
# 回复成训练集和测试集
x_train = data_all[:len_train]
x_test = data_all[len_train:]

print("tf-idf 处理结束")

# 使用GBDT进行训练
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, max_depth=3)
#
# # 配置超参数的搜索组合
# params = {"n_estimators": [400], "learning_rate": [0.1],
#           "max_depth": [4]}
# gs = GridSearchCV(gbdt, params, cv=4, n_jobs=-1, verbose=1)

gbdt.fit(x_train, y_train)

# print(gs.best_params_)
# print(gs.best_score_)
#
# from sklearn.cross_validation import cross_val_score
# print("gbdt10折交叉得分",
#       np.mean(cross_val_score(gbdt, x_train, y_train, cv=10, scoring='roc_auc')))


test_predicted = np.array(gbdt.predict(x_test))


print('保存结果...')
gbdt_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
gbdt_output['id'] = test['id']
gbdt_output = gbdt_output[['id', 'sentiment']]
gbdt_output.to_csv('gbdt_output.csv', index=False)
print('结束.')