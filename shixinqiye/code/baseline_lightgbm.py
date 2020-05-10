#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import category_encoders as ce
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split


submission = pd.read_csv('./data/submission.csv')
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
train_label = pd.read_csv('./data/train_label.csv')

train.head()

np.where(train.isnull().sum()/train.shape[0]<0.5 )[0]
train.columns[np.where(train.isnull().sum()/train.shape[0]<0.5 )[0]]

columns = ['ID', '企业类型', '登记机关', '企业状态', '邮政编码', '注册资本', '核准日期', '行业代码', '经营期限自',
       '成立日期', '行业门类', '企业类别', '管辖机关', '经营范围', '增值税', '企业所得税', '印花税', '教育费',
       '城建税']

train[columns].isnull().sum()/train.shape[0]

train['经营范围'].map(lambda x: len(x))

feature = [ '企业类型', '登记机关', '企业状态', '注册资本', '行业代码', 
        '行业门类', '企业类别', '管辖机关', '经营范围', '增值税', '企业所得税', '印花税', '教育费',
       '城建税']

train[feature].head()

train = train.merge(train_label,on='ID',how = 'left')

data = train.append(test)

data['经营范围'] = data['经营范围'].map(lambda x : len(x))

object_col = ['企业类型','行业门类', '企业类别', '管辖机关']
for i in tqdm(object_col):
    lbl = LabelEncoder()
    data[i] = lbl.fit_transform(data[i].astype(str))
    data[i] = data[i]

data['企业所得税与增值税之比'] = data['增值税']/data['企业所得税']

feature += ['企业所得税与增值税之比']

tr_index = ~data['Label'].isnull()
train = data[tr_index].reset_index(drop=True)
y = data[tr_index]['Label'].reset_index(drop=True).astype(int)
test = data[~tr_index].reset_index(drop=True)
print(train.shape,test.shape)


from sklearn.metrics import roc_auc_score


def lgb_roc_auc_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) 
    return 'f1', roc_auc_score(y_true, y_hat), True

lgb_paras = {'objective': 'binary',
             'learning_rate': 0.1 ,
             'max_depth': 6 ,
             'feature_fraction': 0.8, 
             'bagging_fraction' : 0.8,
             'num_threads':-1}


fi = []
cv_score = []
test_pred = np.zeros((test.shape[0],))
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)

from xgboost import XGBRegressor

for index, (train_index, test_index) in enumerate(skf.split(train, y)):
    print(index)
    train_x, test_x, train_y, test_y = train.iloc[train_index],train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    lgb_model = lgb.train(lgb_paras,
                          train_set = lgb.Dataset(train_x[feature], train_y),
                          valid_sets=[lgb.Dataset(test_x[feature],test_y)],
                          num_boost_round=800 ,
                          feval=lgb_roc_auc_score,
                          verbose_eval=100,
                          categorical_feature = object_col, 
                          )
    y_val = lgb_model.predict(test_x[feature])
    print( "roc_auc:", roc_auc_score(test_y , y_val) )
    cv_score.append(roc_auc_score(test_y,y_val))
    print("cv_score:", cv_score[index])
    test_pred += lgb_model.predict(test[feature]) / 5

submission['Label'] = test_pred
submission.to_csv('submission_light_gbm.csv',index=False)
