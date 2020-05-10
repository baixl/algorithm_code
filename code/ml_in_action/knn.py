#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: knn
date: 2017/10/22
bref:机器学习实战 第2章：knn
"""

import operator
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def create_data_set():
    """
    创建用于knn demo的数据集
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, data_set, labels, k):
    """
    使用create_data_set数据测试knn
    :param inx: 用于确定分类的输入数据
    :param data_set: 输入的训练数据集
    :param labels: 训练数据集的标签
    :param k: 最近邻的数目
    :return:
    """
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(inx, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)  # axis=1:按行相加，axis=0：按列相加
    distances = sq_distance ** 0.5
    # print distances
    sorted_distindices = distances.argsort()  # numpy 中的快速排序，返回元素下标  sorted_distindices= [2 3 1 0]
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distindices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file_to_matrix(file_name):
    """
    将输入数据集转化成矩阵
    :param file_name:
    :return:
    """
    fr = open(file_name)
    lines = fr.readlines()
    numbers_lines = len(lines)
    return_mat = np.zeros((numbers_lines, 3))
    class_label_vector = []
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split("\t")
        return_mat[index, :] = list_from_line[0:3]
        index += 1
        class_label_vector.append(int(list_from_line[-1]))
    return return_mat, class_label_vector


def plg_data(data_mat, label_data):
    """

    :param data_mat: 数据矩阵
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 1], data_mat[:, 2], 15.0 * np.array(label_data), 15.0 * np.array(label_data))
    plt.xlabel(u'游戏时间')
    plt.ylabel(u'消耗冰激凌数')
    plt.show()


def auto_norm(data_set):
    """
    数据归一化
    :param data_set:
    :return:
    """
    min_val = data_set.min(0)  # 0:从列中选择最小值，1：从行中选择最小值
    max_val = data_set.max(0)
    _range = max_val - min_val
    norm_data = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data = data_set - np.tile(min_val, (m, 1))
    norm_data = norm_data / np.tile(_range, (m, 1))
    return norm_data, _range, min_val


def dataing_class_test():
    """
    测试分类器的错误率
    :return:
    """
    return_mat, dating_labels = file_to_matrix("./data_sets/datingTestSet2.txt")
    # plg_data(return_mat, dating_labels)
    norm_data, _ranges, min_vals = auto_norm(return_mat)
    horatio = 0.1
    m = norm_data.shape[0]  # 返回数据量的大小
    num_test_vecs = int(m * horatio)  # 测试集大小
    print num_test_vecs
    err_count = 0.0  # 错误率
    for i in range(num_test_vecs):
        classifierResult = classify0(norm_data[i, :], norm_data[num_test_vecs:m, :],
                                     dating_labels[num_test_vecs:m], 5)
        # print "the classifier came back with: %d, the real answeris : %d" \
        #       % (classifierResult, dating_labels[i])
        if (classifierResult != dating_labels[i]):
            err_count += 1.0
            print "the classifier came back with: %d, the real answeris : %d" \
                  % (classifierResult, dating_labels[i])
    print "the total err rate is : %f" % (err_count / float(num_test_vecs))


def img_to_vec(file_names):
    """
    手写数字书别，将图片的数据转化长numpy矩阵
    将32x32的图片转化成 1x1024
    :param file_names:
    :return:
    """
    return_vec = np.zeros((1, 1024))
    fr = open(file_names)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0, 32 * i + j] = int(line_str[j])
    return return_vec


def handing_writing_digit_clsss_test():
    """
    knn实现手写数字识别
    :return:
    """
    hwlabels = []
    training_files_list = listdir('./data_sets/trainingDigits')
    training_files_nums = len(training_files_list)  # 训练集的文件数目
    training_mats = np.zeros((training_files_nums, 1024))  # 将每个文件读成一行：1 x 1024
    for i in range(training_files_nums):
        file_name = training_files_list[i]
        file_str = file_name.split('.')[0]
        class_num_str = int(file_str.split('_')[0])  # 类别标记
        hwlabels.append(class_num_str)
        training_mats[i, :] = img_to_vec('./data_sets/trainingDigits/%s' % file_name)
    test_files_list = listdir('./data_sets/testDigits')
    err_count = 0.0
    test_nums = len(test_files_list)
    for i in range(test_nums):
        file_name = test_files_list[i]
        file_str = file_name.split('.')[0]
        class_num_str = int(file_str.split('_')[0])  # 类别标记
        vec_under_test = img_to_vec('./data_sets/testDigits/%s' % file_name)
        classifier_result = classify0(vec_under_test, training_mats, hwlabels, 5)

        if classifier_result != class_num_str:
            err_count += 1.0
            print "the classifier came back with: %d, the real answeris : %d" \
                  % (classifier_result, class_num_str)
    print "the total number of errors is : %d" % err_count
    print "the total error rate is : %f" % (err_count / float(test_nums))


if __name__ == '__main__':
    """
    入口文件
    """
    # group, labels = create_data_set()
    # print classify0([0, 0], group, labels, 3)

    # datingTestSet2.txt 是datingTestSet.txt离散化,分别是：每年获得飞行里程数，玩游戏所耗费时间百分比，每周消费冰淇淋公升数，喜好程度（1，2，3）
    return_mat, dating_labels = file_to_matrix("./data_sets/datingTestSet2.txt")
    # plg_data(return_mat, dating_labels)
    norm_data, _ranges, min_vals = auto_norm(return_mat)

    file_name = './data_sets/testDigits/0_13.txt'
    test_vec = img_to_vec(file_name)
    print test_vec[0, 0:31]

    handing_writing_digit_clsss_test()
