#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: pca
date: 2017/12/9
bref:
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float, line) for line in stringArr]
    return np.mat(dataArr)


def pca(dataMat, topNfeat=99999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    covMat = np.cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 特征分解 eigVals 特征值，eigVects 特征向量

    eigValInd = np.argsort(eigVals)  # 从小到大对特征值排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 选择最大的N个
    redEigVects = eigVects[:, eigValInd]

    lowDDataMat = meanRemoved * redEigVects  # 将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = loadDataSet('./data_sets/testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print np.shape(lowDMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c="red")
    plt.show()
