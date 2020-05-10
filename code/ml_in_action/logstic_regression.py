#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: logstic_regression
date: 2017/11/5
bref:机器学习实战第5章，Logistic回归
"""
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    """

    :return:
    """
    data_mat = []
    label_mat = []
    # 该数据集有x1 x2连个特征，第3个值是分类类别
    fr = open('./data_sets/testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmod(x):
    """
    sigmod 函数
    :return:
    """
    return 1.0 / (1 + np.exp(-x))


def gradAscent(data_mat_in, class_lables):
    """
    梯度上升算法
    :param data_mat_in:
    :param class_lables:
    :return:
    """
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_lables).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))  # 初始化所有权重为1
    for k in range(max_cycles):
        h = sigmod(data_matrix * weights)
        error = label_mat - h  # 真实类别与预测值的差别
        weights = weights + alpha * data_matrix.transpose() * error  # 沿着差值的方向调整系数
    return weights


def stoGradAscent0(data_matrix, class_lables):
    """
    随机梯度上升算法,一次选择一个样本进行权重更新
    :param data_matrix:
    :param class_lables:
    :return:
    """
    m, n = np.shape(data_matrix)  # m 样本空间大小，n样本特征数
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmod(np.sum(data_matrix[i] * weights))
        error = class_lables[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stoGradAscent1(data_matrix, class_lables, num_iter=150):
    """
    随机梯度上升算法,一次选择一个样本进行权重更新,
    :param data_matrix:
    :param class_lables:
    :param num_iter 迭代次数
    :return:
    """
    m, n = np.shape(data_matrix)  # m 样本空间大小，n样本特征数
    alpha = 0.01
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))

            h = sigmod(np.sum(data_matrix[rand_index] * weights))
            error = class_lables[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
    return weights


def plot_best_fit(weights):
    """
    画出决策拟合的边界
    :param weights:
    :return:
    """
    data_mat, label_mat = load_data()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])  # 特征1
            ycord1.append(data_arr[i, 2])  # 特征2
            print data_arr[i, 1]
            data_arr[i, 1]
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def classify_vector(inx, weights):
    """
    
    :param inx: 
    :param weights: 
    :return: 
    """
    prob = sigmod(np.sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    """

    :return:
    """
    fr_train = open('./data_sets/horseColicTraining.txt')
    fr_test = open('./data_sets/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(current_line[21]))
    training_weights = stoGradAscent1(np.array(training_set), training_labels, 1000)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        if int(classify_vector(np.array(line_arr), training_weights)) != int(current_line[21]):
            error_count += 1
    error_rate = float(error_count) / num_test_vec
    print "the error rate of this test is : %f" % error_rate
    return error_rate


def multi_test():
    num_tests = 100
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print "after %d iterations  the average errr rate is : %f" % (num_tests, error_sum / float(num_tests))


if __name__ == '__main__':
    """
    入口函数
    """
    data_arr, label_mat = load_data()
    weights = gradAscent(data_arr, label_mat)
    # plot_best_fit(weights.getA())  # matrix 转化成 ndarray
    #
    # weight_sto_gard = stoGradAscent0(np.array(data_arr), label_mat)
    # plot_best_fit(weight_sto_gard)
    # 优化过的随机梯度算法
    # weight_sto_gard2 = stoGradAscent1(np.array(data_arr), label_mat, 2000)
    # plot_best_fit(weight_sto_gard2)
    multi_test()
