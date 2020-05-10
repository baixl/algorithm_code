#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: decisionTrees
date: 2017/11/4
bref: 机器学习实战 决策树练习
"""

import math
import operator


def demo_data():
    """
    测试数据集
    :return:
    """
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    label = ['no surfacing', 'flippers']
    return data_set, label


def calc_shannon_ent(data_set):
    """

    计算给定数据集的香农熵
    熵描述了一个变量包含信息量的大小,熵越高，则信息的不纯度越高
    :param data_set:
    :return: 以2为底的对数

    """
    num_entries = len(data_set)
    label_counts = {}
    for feature_vec in data_set:
        current_label = feature_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 1
        else:
            label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def split_dta(data_set, axis, value):
    """

    :param data_set:
    :param axis:
    :param value:
    :return:
    """
    # todo 这个分割的细节没看懂
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


def choose_best_feature_tosplit(data_set):
    """
    通过按特征划分数据集，一次计算信息熵
    选择是信息增益最大的特征作为划分特征
    :param data_set:
    :return:
    """
    feature_nums = len(data_set[0]) - 1  # 原始数据集的特征数目
    base_entropy = calc_shannon_ent(data_set)  # 原始数据的信息熵
    best_info_gain = 0.0  # 信息增益
    best_feature_index = -1  # 选择的最好特征的下标
    for i in range(feature_nums):

        feature_list = [feature[i] for feature in data_set]  # 获取当前特征所在的列
        uniquevals = set(feature_list)
        new_entropy = 0.0
        print uniquevals
        for val in uniquevals:
            sub_data_set = split_dta(data_set, i, val)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = i
    return best_feature_index


def majority_cnt(class_list):
    """
    :param class_list:
    :return:投票最多的类别标签
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 1
        else:
            class_count += 1
    sorted_class = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)  # 按值倒序排列
    return sorted_class[0][0]


def create_tree(data_set, labels):
    """

    :param data_set:
    :param labels:
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        # 类别完全相同, 停止继续划分
        return class_list[0]
    if len(data_set[0]) == 1:
        # 遍历完所有特征时，返回出现次数最多的类别
        return majority_cnt(class_list)
    best_feature = choose_best_feature_tosplit(data_set)  # 最好的特征的下标
    best_feature_label = labels[best_feature]
    tree = {best_feature: {}}
    del (labels[best_feature])
    feature_vals = [example[best_feature] for example in data_set]
    unique_vals = set(feature_vals)
    for val in unique_vals:
        sub_labels = labels[:]
        tree[best_feature][val] = create_tree(split_dta(data_set, best_feature, val), sub_labels)
    return tree


if __name__ == '__main__':
    """
    入口函数
    """
    my_data, labels = demo_data()
    # print calc_shannon_ent(my_data)

    # print split_dta(my_data, 0, 0)
    # choose_best_feature_tosplit(my_data)

    tree = create_tree(my_data, labels)
    print tree
