# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
author: xiaolingbai@gmail.com
file: titanic
date: 2017/8/20
bref: 坦坦尼克游客生还， 决策树
"""

import pandas as pd

# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt
titanic = pd.read_csv('./titanic.txt')

# 观察前几行数据
print titanic.head()
print titanic.info()

# 特征选择，暂且选择，仓位等级，年龄，性别
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 分析当前的特征
print x.info()
# 结果，age只有633个
# pclass 和 sex 是object类型，需要转化成数字，比如0/1

# 使用平均数替换掉age中的缺省值
x['age'].fillna(x['age'].mean(), inplace=True)
print x.info()

# 数据分割

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.24, random_state=33)

# 特征转换
from sklearn.feature_extraction import DictVectorizer

