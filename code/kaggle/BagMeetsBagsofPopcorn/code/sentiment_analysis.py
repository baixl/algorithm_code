#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo
date: 2017/11/29
bref:
"""
import pandas as pd
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words(raw_review):
    """
    数据预处理：提取、清晰、分词
    :param raw_review:
    :return:
    """
    # 去除html标签
    review_text = BeautifulSoup(raw_review).get_text()
    # 去除非字母
    letter_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 转化成小写字母
    words = letter_only.lower().split()
    stops = set(stopwords.words('english'))
    # 去除停用词
    meaningful_words = [w for w in words if not w in stops]

    return " ".join(meaningful_words)


def read_data():
    train = pd.read_csv('../input/labeledTrainData.tsv', header=0, delimiter='\t')
    test = pd.read_csv('../input/testData.tsv', header=0, delimiter='\t')
    label = train['sentiment']
    train_data = []
    for i in range(len(train['review'])):
        train_data.append(review_to_words(train['review'][i]))
    test_data = []
    for i in range(len(test['review'])):
        train_data.append(review_to_words(test['review'][i]))
    return train_data, test_data


def tf_idf_handle(train_data, test_data):
    """
    tf-idf 向量化
    :param train_data:
    :param test_data:
    :return:
    """
    tfidf = CountVectorizer(min_df=2,  # 最小支持度
                            max_features=None,
                            strip_accents='unicode',
                            analyzer='word',
                            token_pattern=r'\w{1,}',
                            ngram_range=(1, 3),  # 二元文法模型
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            stop_words='english')  # 去掉英文停用词
    data_all = test_data + test_data
    len_train = len(train_data)
    tfidf.fit(data_all)
    data_all = tfidf.transform(data_all)

    # 将训练集和测试集通过TF-IDF向量化
    train_x = data_all[: len_train]
    test_x = data_all[len_train:]

    print train_x[0]
    print "tf-idf handle done "


if __name__ == '__main__':
    train = pd.read_csv('../input/labeledTrainData.tsv', header=0, delimiter='\t')

    clean_train_reviews = []
    nums = train['review'].size
    for i in xrange(0, nums):
        if (i + 1) % 1000 == 0:
            print "review %d of %d \n" % (i + 1, nums)
        clean_train_reviews.append(review_to_words(train['review'][i]))

    # 使用scikit-learn创建文本的词袋模型
    vector = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=500)

    # fit_transform 执行两个操作：1 拟合模型和学习词袋模型  2 将训练数据转化为特征向量
    train_data_features = vector.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    print train_data_features.shape

    vocab = vector.get_feature_names()
    print vocab
    print train_data_features[0]

    # dist = np.sum(train_data_features, axis=0)
    #
    # for tag, count in zip(vocab, dist):
    #     print tag, count

    # 使用有100个决策树的随机森林进行训练
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(train_data_features, train['sentiment'])

    # 处理训练数据
    test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify that there are 25,000 rows and 2 columns
    print 'test shape:', test.shape

    # Create an empty list and append the clean reviews one by one
    num_reviews_test = len(test["review"])
    clean_test_reviews = []

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0, num_reviews_test):
        if ((i + 1) % 1000 == 0):
            print "Review %d of %d\n" % (i + 1, num_reviews_test)
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vector.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    result = forest.predict(test_data_features)
    output = pd.DataFrame(data={'id': test['id'], "snetiment": result})
    output.to_csv('../input/bag_of_words_model_result.csv', index=False, quoting=3)
