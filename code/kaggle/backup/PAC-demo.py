# -*- coding:utf-8 -*-
# !/usr/bin/env python
######

"""
author: xiaolingbai@gmail.com
file: PAC-demo
date: 2017/8/22
bref: 手写数字识别
"""

import pandas as pd
import numpy as np

digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

x_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

from sklearn.decomposition import PCA
# 初始化一个将高维压缩至2维的PCA
estimator = PCA(n_components = 2)
x_pca = estimator.fit_transform(x_digits)

# 显示10类手写字体进过压缩后二维的空间分布
from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan','orange', 'gray']
    for i in xrange(len(colors)):
        px = x_pca[:, 0][y_digits.as_matrix() ==i]
        py = x_pca[:, 1][y_digits.as_matrix() ==i]
        plt.scatter(px, py, c = colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.show()
plot_pca_scatter()