#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: if_idf_sentiment_analysis
date: 2017/12/16
bref:
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    train = pd.read_csv('../input/labeledTrainData.csv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../input/testData.csv', header=0, delimiter="\t", quoting=3)
    label = train['sentiment']

    train_data = []
    for i in range(len(train['review'])):
        train_data.append(train['review'][i])
    test_data = []
    for i in range(len(test['review'])):
        test_data.append(test['review'][i])

    # 使用TF-IDF向量化
    tfidf = CountVectorizer(min_df=1,  # 最小支持度
                            max_features=None,
                            strip_accents='unicode',
                            analyzer='word',
                            token_pattern=r'\w{1,}',
                            ngram_range=(3, 3),  # 二元文法模型
                            stop_words='english')  # 去掉英文停用词
    data_all = train_data + test_data
    len_train = len(train_data)
    tfidf.fit(data_all)
    data_all = tfidf.transform(data_all)
    # 将训练集和测试集通过TF-IDF向量化
    train_x = data_all[: len_train]
    test_x = data_all[len_train:]
    print "tf-idf handle done "

    # 使用朴素贝叶斯预测
    model_NB = MultinomialNB()
    model_NB.fit(train_x, label)

    print "多项式贝叶斯分类器和10折交叉验证结果：", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))

    test_prediction = np.array(model_NB.predict(test_x))
    print "保存结果："
    nb_output = pd.DataFrame(data=test_prediction, columns=['sentiment'])
    nb_output['id'] = test['id']
    nb_output = nb_output[['id', 'sentiment']]
    nb_output.to_csv('../nb_output.csv', index=False)
    print "处理结束"
