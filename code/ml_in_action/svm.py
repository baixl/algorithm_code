#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: svm
date: 2017/11/9
bref:机器学习实战 svm一章的联系，纯手码

"""
import numpy as np
import random


def load_data_set(file_name):
    """
    准备数据
    :param file_name:
    :return:
    """
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_jrand(i, m):
    """
    选择一个不等于i的随机数，大小为从0到m的整数
    :param i: 第1个alpha的下标
    :param m:alpha的数目
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i:]).T) + b
        ei = fxi - float(label_mat[i])
        if (label_mat[i] * ei < -toler and alphas[i] < C) or (label_mat[i] * ei > -toler and alphas[i] > C):
            j = select_jrand(i, m)
            fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j:]).T) + b
            ej = fxj - float(label_mat[j])
            alphas_i_old = alphas[i].copy()
            alphas_j_old = alphas[j].copy()
            if label_mat[i] != label_mat[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                print 'L==H'
                continue




if __name__ == '__main__':
    data_arr, label_arr = load_data_set('./data_sets/testSet_svm.txt')
    print label_arr
