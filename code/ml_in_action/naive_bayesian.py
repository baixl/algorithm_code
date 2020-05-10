#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: naive_bayesian
date: 2017/11/5
bref:机器学习实战第4章，朴素贝叶斯分类器
"""
import numpy as np
import math


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    创建一个包含文档所有单词的词表
    :param data_set:
    :return:
    """
    vocab_set = set([])  # create empty set
    for document in data_set:
        vocab_set = vocab_set | set(document)  # union of the two sets
    return list(vocab_set)


def set_of_words_2vec(vocab_list, input_set):
    """
    根据词表，和输入文档，使用one-hot 创建文档对应的向量
    出现在词表中：1，否则：0
    :param vocab_list:
    :param input_set:
    :return:文档向量
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return return_vec


def train_NB0(train_matrix, train_category):
    """
    避免概率为0和条件概率相乘出现0：
    1 初始化分子为1，分母为2
    2 使用对数代码特征的概率
    :param train_matrix: 文档矩阵
    :param train_category: 文档对应的类别向量
    :return:
    """
    num_train_docs = len(train_matrix)  # 文档数量
    num_words = len(train_matrix[0])  #
    # train_category 是 0 1标记，计算1的总数就可以算出abusive的类别数
    p_abusive = np.sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)  # 避免某个词未出现，而出现概率值为0
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        # 遍历每个文档
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += np.sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += np.sum(train_matrix[i])

    # 使用对数避免多个值很小的浮点数相乘出现下溢，详见<ml in action> p62
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)

    return p0_vec, p1_vec, p_abusive


def classifyNB(vec_2classify, p0_vec, p1_vec, pclass):
    """

    :param vec_2classify:
    :param p0_vec:
    :param p1_vec:
    :param pclass:
    :return:
    """
    p1 = np.sum(vec_2classify * p1_vec) + math.log(pclass)
    p0 = np.sum(vec_2classify * p0_vec) + math.log(1.0 - pclass)
    if p1 > p0:
        return 1
    else:
        return 0


def test_NB():
    """

    :return:
    """
    list_posts, list_class = load_data_set()
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post in list_posts:
        train_mat.append(set_of_words_2vec(my_vocab_list, post))
    p0v, p1v, pab = train_NB0(np.array(train_mat), np.array(list_class))
    test_entry = ['my', 'stupid']
    this_vec = np.array(set_of_words_2vec(my_vocab_list, test_entry))
    print test_entry, 'classified is :', classifyNB(this_vec, p0v, p1v, pab)


def bag_of_word_2vecNN(vocab_list, input_set):
    """
    词袋模型
    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


if __name__ == '__main__':
    """
    入口函数
    """
    list_posts, list_class = load_data_set()
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post in list_posts:
        train_mat.append(set_of_words_2vec(my_vocab_list, post))
    print train_mat

    test_NB()
