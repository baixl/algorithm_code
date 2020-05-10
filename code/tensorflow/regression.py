#/us/bin/python3
#-*- coding:utf-8 -*-
# @FileName: regression.py
# @Author:baixiaoling
# @Time:2018/11/25 22:28
# tensorflow regression， 波士顿房价预测


import tensorflow as tf
from tensorflow import keras

import numpy as np

# 波士顿房价数据集
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
# 打散数据集
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
