#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: prediction
date: 2018/3/29
bref:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# 读取数据
train_data_df = pd.read_csv("./input/train.csv", index_col=0)
test_data_df = pd.read_csv("./input/test.csv", index_col=0)
# 取出价格数据， 合并数据，然后预处理

prices = pd.DataFrame({"price": train_data_df['SalePrice'], "log(1+price)": np.log1p(train_data_df['SalePrice'])})
y_train = np.log1p(train_data_df.pop('SalePrice'))
all_df = pd.concat((train_data_df, test_data_df), axis=0)
print all_df['MSSubClass'].dtypes
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
print all_df['MSSubClass'].dtypes
print all_df['MSSubClass'].value_counts()

print pd.get_dummies(all_df['MSSubClass'],prefix = 'MSSubClass').head()
all_dummy_df = pd.get_dummies(all_df)
# print all_dummy_df.head()


print all_dummy_df.isnull().sum().sort_values(ascending = False).head(11)
# 我们这里用mean填充
mean_cols = all_dummy_df.mean()
print mean_cols.head(10)
all_dummy_df = all_dummy_df.fillna(mean_cols)
print all_dummy_df.isnull().sum().sum()

# 标准化numerical数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print numeric_cols
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std
# 把数据处理之后，送回训练集和测试集
dummy_train_df = all_dummy_df.loc[train_data_df.index]
dummy_test_df = all_dummy_df.loc[test_data_df.index]
print dummy_train_df.shape,dummy_test_df.shape

# 将DF数据转换成Numpy Array的形式，更好地配合sklearn

X_train = dummy_train_df.values
X_test = dummy_test_df.values

