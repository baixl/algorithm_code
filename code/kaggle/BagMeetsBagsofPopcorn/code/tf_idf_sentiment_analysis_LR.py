#!/usr/bin/env python3
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
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    train = pd.read_csv('../input/clean_labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../input/clean_testData.tsv', header=0, delimiter="\t", quoting=3)
    label = train['sentiment']

    train_data = []
    for i in range(len(train['review'])):
        train_data.append(train['review'][i])
    test_data = []
    for i in range(len(test['review'])):
        test_data.append(test['review'][i])

    # 使用TF-IDF向量'
    tfidf = TfidfVectorizer(min_df=3,
                            max_features=None,
                            strip_accents='unicode',
                            analyzer='word',
                            token_pattern=r'\w{1,}',
                            ngram_range=(1, 2),
                            use_idf=1,
                            sublinear_tf=1,
                            stop_words='english', )

    data_all = train_data + test_data
    len_train = len(train_data)
    tfidf.fit(data_all)
    data_all = tfidf.transform(data_all)
    # 将训练集和测试集通过TF-IDF向量化
    train_x = data_all[: len_train]
    test_x = data_all[len_train:]
    print("tf-idf handle done ")

    # 尝试使用卡方检验
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    chi2mode = SelectKBest(chi2, k=200000)
    train_x = chi2mode.fit_transform(train_x, label)
    test_x = chi2mode.transform(test_x)
    print("卡方特征提取结束")

    print("使用但模型LR训练")
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import GridSearchCV
    from sklearn.cross_validation import cross_val_score

    model_LR = LR(penalty='l2', dual=True, tol=0.0001,
                  C=1, fit_intercept=True, intercept_scaling=1.0,
                  class_weight=None, random_state=None)

    print("20 Fold CV Score: ", np.mean(cross_val_score(model_LR, test_x, label, cv=20, scoring='roc_auc')))
    # grid_values = {"C": [30]}
    # model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
    model_LR.fit(train_x, label)
    # 20折交叉验证
    # gsc =GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True,
    #                                  fit_intercept=True, intercept_scaling=1, penalty='L2', random_state=0, tol=0.0001),
    #              fit_params={}, iid=True, n_jobs=1,
    #              param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
    #              scoring='roc_auc', verbose=0)

    # gsc.fit(train_x, label)
    # print(model_LR.best_params_)
    # print(model_LR.best_score_)
    # print(model_LR.cv_results_)


    test_prediction = np.array(model_LR.predict(test_x))
    print("保存结果：")
    lr_output = pd.DataFrame(data=test_prediction, columns=['sentiment'])
    lr_output['id'] = test['id']
    lr_output = lr_output[['id', 'sentiment']]
    lr_output.to_csv('../result/output_LR_chi2.csv', index=False)
    print("处理结束")
