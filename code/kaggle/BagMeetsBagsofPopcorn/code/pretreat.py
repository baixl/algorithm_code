#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: pre_handle_data
date: 2017/12/16
bref:数据预处理
"""

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


def review_to_words(raw_review):
    """
    数据预处理：提取、清晰、分词
    :param raw_review:
    :return:
    """
    # 去除html标签
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    # 去除非字母
    letter_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 转化成小写字母
    words = letter_only.lower().split()
    stops = set(stopwords.words('english'))
    # 去除停用词
    meaningful_words = [w for w in words if not w in stops]

    return " ".join(meaningful_words)


if __name__ == '__main__':
    # file1_in = '../input/labeledTrainData.tsv'
    # file1_out = "../input/clean_labeledTrainData.tsv"
    # file1 = open(file1_out, 'a+')
    # i=0
    # with open(file1_in, 'rb') as f:
    #     for line in f:
    #         i = i+1
    #         if i%1000 == 0:
    #             print('Review %d \n' % (i + 1, ))
    #         line_arr = line.split("\t")
    #         str = line_arr[0] + "\t" + line_arr[1] + "\t" + review_to_words(line_arr[2]) + "\n"
    #         file1.write(str)
    # file1.close()
    #
    # file2_in = '../input/testData.tsv'
    # file2_out = "../input/clean_testData.tsv"
    # file2 = open(file2_out, 'a+')
    # i=0
    # with open(file2_in, 'rb') as f:
    #     for line in f:
    #         i = i + 1
    #         if i % 1000 == 0:
    #             print('Review %d \n' % (i + 1,))
    #         line_arr = line.split("\t")
    #         str = line_arr[0] + "\t" + review_to_words(line_arr[1]) + "\n"
    #         file2.write(str)
    # file2.close()

    file3_in = '../input/unlabeledTrainData.tsv'
    file3_out = "../input/clean_unlabeledTrainData.tsv"
    file3 = open(file3_out, 'a+')
    i = 0
    with open(file3_in, 'rb') as f:
        for line in f:
            i = i + 1
            if i % 1000 == 0:
                print('Review %d \n' % (i + 1,))
            line_arr = line.split("\t")
            str = line_arr[0] + "\t" + review_to_words(line_arr[1]) + "\n"
            file3.write(str)
    file3.close()
