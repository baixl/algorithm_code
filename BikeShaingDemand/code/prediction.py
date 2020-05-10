#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: prediction
date: 2018/4/9
bref:
"""
import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from datetime import datetime

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()

labels = train['count']
labels = labels.apply(lambda x: np.log1p(x))
import calendar
from datetime import datetime

dailyData = train
dailyData.datetime = dailyData.datetime.apply(pd.to_datetime)
dailyData['month'] = dailyData.datetime.apply(lambda x: x.month)
dailyData['hour'] = dailyData.datetime.apply(lambda x: x.hour)
dailyData['year'] = dailyData.datetime.apply(lambda x: x.year)
dailyData['weekday'] = dailyData.datetime.apply(lambda x: x.weekday())

numericalFeatureNames = ["temp", "humidity", "atemp"]
dropFeatures = ['casual', "count", "datetime", "registered", "windspeed"]
categryVariable = ['season', 'holiday', "workingday", "weather", "month", "hour", "weekday"]

for category in categryVariable:
    dummy = pd.get_dummies(dailyData[category], prefix=category)
    dailyData = dailyData.join(dummy)

dailyData = dailyData.drop(categryVariable, axis=1)
dailyData = dailyData.drop(dropFeatures, axis=1)


# 定义代价函数
def rmsle(y, y_):
    y = np.expm1(y)
    y_ = np.expm1(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    cal = np.square(log1 - log2)
    return np.sqrt(np.mean(cal))


from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(dailyData, labels, test_size=0.235, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

print("开始训练")
# parameters = [{'n_estimators': [100, 400],
#                'max_depth': [10, 20, ],
#                }]
# rf = RandomForestRegressor(random_state=14)
# clf = GridSearchCV(rf, parameters, cv=5)
# clf.fit(train_data, train_labels)
# print(clf.best_params_)
# print(clf.best_score_)
# print(clf.grid_scores_)


parameter_space = [{
    # 'n_estimators': [ 1100, 1200, 1400, 1600],  # 最优1000
    # 'max_depth': [3, 4, 5, 6],
    # 'learning_rate': [0.1, 0.2]
    'subsample':[0.5,0.8]
}]
from  xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate=0.1, n_estimators=1200, max_depth=4, gamma=0, subsample=0.8)
# clf = GridSearchCV(xgb, param_grid=parameter_space, cv=5)
# #
# clf.fit(train_data, train_labels)
#
# print(clf.grid_scores_)
# print(clf.best_params_)
# print(clf.best_score_)
#
xgb.fit(train_data, train_labels)
preds = xgb.predict(test_data)
print("RMSLE Value For XGB Boost: ", rmsle(test_labels, preds))
