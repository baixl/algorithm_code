# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
author: xiaolingbai@gmail.com
file: cancer_prediction
date: 2017/8/19
bref:
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读入数据
df_train = pd.read_csv('./breast-cancer-train.csv')
df_test = pd.read_csv('./breast-cancer-test.csv')

# 将训练集 按照Type的不同分为正负样本，选择 Clump Thickness 和 Cell Size 作为特征，
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 绘制散点图 良性标记为 红色 o, 恶习为黑色x
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o',
            linewidths=1, s=100, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x',
            linewidths=1, s=40, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# 显示图
# plt.show()

# 利用numpy中的random随机采样，直线的截距和系数
# 截距
intercept = np.random.random([1])
print intercept
# 系数
coef = np.random.random([2])
print coef
lx = np.arange(0, 12)
print lx
ly = (-intercept - lx * coef[0]) / coef[1]
print ly
# 绘制随机直线
plt.plot(lx, ly, c='blue')

# 导入逻辑斯谛回归
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# 使用所有条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print 'Test accuracy (10 train sample):', lr.score(df_test[['Clump Thickness', 'Cell Size']]
                                                   , df_test['Type'])

intercept = lr.intercept_
coef = lr.coef_[0, :]
print coef
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='green')
plt.show()

# 绿色线是最终生成的分割线，蓝色是初始随机生成的一条线
