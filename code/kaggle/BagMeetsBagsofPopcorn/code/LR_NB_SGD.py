#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: LR_NB_SGD
date: 2018/4/1
bref: http://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/NLP_Movies.ipynb
分别使用LR 朴素贝叶斯、SGD 拟合数据
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def review_to_wordlist(review):
    '''
    Meant for converting each of the IMDB reviews into a list of words.
    '''
    # First remove the HTML.
    review_text = BeautifulSoup(review).get_text()

    # Use regular expressions to only include words.
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert words to lower case and split them into separate words.
    words = review_text.lower().split()

    # Return a list of words
    return (words)

if __name__ == '__main__':

    train = pd.read_csv('../input/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../input/testData.tsv', header=0, delimiter="\t", quoting=3)

    label = train['sentiment']

    train_data = []
    for i in range(len(train['review'])):
        train_data.append(" ".join(review_to_wordlist(train['review'][i])))
    test_data = []
    for i in range(len(test['review'])):
        test_data.append(" ".join(review_to_wordlist(test['review'][i])))

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

    print(train_x.shape)

    # LR
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.grid_search import GridSearchCV

    grid_values = {'C': [30]}  # Decide which settings you want for the grid search.

    model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0),
                            grid_values, scoring='roc_auc', cv=20)
    model_LR.fit(train_x, label)
    print "LR效果"
    print(model_LR.grid_scores_)
    print(model_LR.best_estimator_)

    # NB
    from sklearn.naive_bayes import MultinomialNB as MNB

    model_NB = MNB()
    model_NB.fit(train_x, label)
    from sklearn.cross_validation import cross_val_score
    import numpy as np

    print "MNB效果"
    print("20 Fold CV Score for Multinomial Naive Bayes: ",
          np.mean(cross_val_score(model_NB, train_x, label, cv=20, scoring='roc_auc')))

    # SGD
    from sklearn.linear_model import SGDClassifier as SGD

    sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]}  # Regularization parameter

    model_SGD = GridSearchCV(SGD(random_state=0, shuffle=True, loss='modified_huber'),
                             sgd_params, scoring='roc_auc',
                             cv=20)  # Find out which regularization parameter works the best.

    model_SGD.fit(train_x, label)  # Fit the model.
    print "SGD效果"
    print(model_SGD.grid_scores_)

    # 分别输出 LR、MNB、SGD 的结果
    # LR_result = model_LR.predict_proba(test_x)[:,1]  # We only need the probabilities that the movie review was a 7 or greater.
    LR_result = model_LR.predict(test_x)  # We only need the probabilities that the movie review was a 7 or greater.
    LR_output = pd.DataFrame(
        data={"id": test["id"], "sentiment": LR_result})  # Create our dataframe that will be written.
    LR_output.to_csv('../result/Logistic_Reg_Proj2.csv', index=False,
                     quoting=3)  # Get the .csv file we will submit to Kaggle.
    # Repeat this for Multinomial Naive Bayes


    MNB_result = model_NB.predict(test_x)
    MNB_output = pd.DataFrame(data={"id": test["id"], "sentiment": MNB_result})
    MNB_output.to_csv('../result/MNB_Proj2.csv', index=False, quoting=3)

    # Last, do the Stochastic Gradient Descent model with modified Huber loss.

    SGD_result = model_SGD.predict(test_x)
    SGD_output = pd.DataFrame(data={"id": test["id"], "sentiment": SGD_result})
    SGD_output.to_csv('../result/SGD_Proj2.csv', index=False, quoting=3)
