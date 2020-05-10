# https://github.com/hallochen/jetcompetition/blob/master/baseline.py
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
def count_column(df,column):
    tp = df.groupby(column).count().reset_index()
    tp = tp[list(tp.columns)[0:2]]
    tp.columns = [column, column+'_count']
    df=df.merge(tp,on=column,how='left')
    return df
def count_mean(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['mean']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_mean']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_count(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_count']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_sum(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_sum']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_std(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['std']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_std']
    df = df.merge(tp, on=base_column, how='left')
    return df


train=pd.read_csv('./jet_simple_data/simple_train_R04_jet_10000.csv')
test=pd.read_csv('./jet_simple_data/simple_test_R04_jet_10000.csv')

def energy(df):
    x=df['jet_px']
    y=df['jet_py']
    z= df['jet_pz']
    return (x**2+y**2+z**2)**0.5
train['energy']=train.apply(energy,axis=1)
test['energy']=test.apply(energy,axis=1)

train['x_n']=train['jet_px']/train['energy']
train['y_n']=train['jet_py']/train['energy']
train['z_n']=train['jet_pz']/train['energy']

test['x_n']=test['jet_px']/test['energy']
test['y_n']=test['jet_py']/test['energy']
test['z_n']=test['jet_pz']/test['energy']

train=count_mean(train,'event_id','x_n')
train=count_sum(train,'event_id','x_n')
train=count_std(train,'event_id','x_n')

train=count_mean(train,'event_id','y_n')
train=count_sum(train,'event_id','y_n')
train=count_std(train,'event_id','y_n')

train=count_mean(train,'event_id','z_n')
train=count_sum(train,'event_id','z_n')
train=count_std(train,'event_id','z_n')


test=count_mean(test,'event_id','x_n')
test=count_sum(test,'event_id','x_n')
test=count_std(test,'event_id','x_n')

test=count_mean(test,'event_id','y_n')
test=count_sum(test,'event_id','y_n')
test=count_std(test,'event_id','y_n')

test=count_mean(test,'event_id','z_n')
test=count_sum(test,'event_id','z_n')
test=count_std(test,'event_id','z_n')


train['abs']=train['jet_energy']-train['energy']
test['abs']=test['jet_energy']-test['energy']


train=count_mean(train,'event_id','number_of_particles_in_this_jet')
train=count_sum(train,'event_id','number_of_particles_in_this_jet')
train=count_std(train,'event_id','number_of_particles_in_this_jet')

train=count_mean(train,'event_id','jet_mass')
train=count_sum(train,'event_id','jet_mass')
train=count_std(train,'event_id','jet_mass')

train=count_mean(train,'event_id','jet_energy')
train=count_sum(train,'event_id','jet_energy')
train=count_std(train,'event_id','jet_energy')

train['mean_energy']=train['jet_energy']/train['number_of_particles_in_this_jet']
train['mean_jet_mass']=train['jet_mass']/train['number_of_particles_in_this_jet']
train=count_mean(train,'event_id','mean_energy')
train=count_sum(train,'event_id','mean_energy')
train=count_std(train,'event_id','mean_energy')
train=count_mean(train,'event_id','mean_jet_mass')
train=count_sum(train,'event_id','mean_jet_mass')
train=count_std(train,'event_id','mean_jet_mass')
train=count_mean(train,'event_id','abs')
train=count_sum(train,'event_id','abs')
train=count_std(train,'event_id','abs')
train=count_mean(train,'event_id','energy')
train=count_sum(train,'event_id','energy')
train=count_std(train,'event_id','energy')

test=count_mean(test,'event_id','number_of_particles_in_this_jet')
test=count_sum(test,'event_id','number_of_particles_in_this_jet')
test=count_std(test,'event_id','number_of_particles_in_this_jet')

test=count_mean(test,'event_id','jet_mass')
test=count_sum(test,'event_id','jet_mass')
test=count_std(test,'event_id','jet_mass')

test=count_mean(test,'event_id','jet_energy')
test=count_sum(test,'event_id','jet_energy')
test=count_std(test,'event_id','jet_energy')

test['mean_energy']=test['jet_energy']/test['number_of_particles_in_this_jet']
test['mean_jet_mass']=test['jet_mass']/test['number_of_particles_in_this_jet']
test=count_mean(test,'event_id','mean_energy')
test=count_sum(test,'event_id','mean_energy')
test=count_std(test,'event_id','mean_energy')
test=count_mean(test,'event_id','mean_jet_mass')
test=count_sum(test,'event_id','mean_jet_mass')
test=count_std(test,'event_id','mean_jet_mass')
test=count_mean(test,'event_id','abs')
test=count_sum(test,'event_id','abs')
test=count_std(test,'event_id','abs')
test=count_mean(test,'event_id','energy')
test=count_sum(test,'event_id','energy')
test=count_std(test,'event_id','energy')

train=train.drop_duplicates(subset=['event_id']).reset_index(drop=True)
# train=train.sort_values(by='event_id').reset_index(drop=True)

d={1:[1,0,0,0,],4:[0,1,0,0],5:[0,0,1,0],21:[0,0,0,1]}
def label_process(x):
    x=d[x]
    return x
train['label']=train['label'].apply(label_process)
train_y=train.pop('label').values
train_y=np.array(list(train_y))
_=train.pop('jet_id')
test_id=test.pop('jet_id')
_=train.pop('event_id')
_=test.pop('event_id')
train_x=train.values
test_x=test.values

#######
train_y=train_y.argmax(axis=1)
print(train_x.shape,test_x.shape,train_y.shape)
print(train.columns)

results = np.zeros((len(test_x), 4), dtype='float')
kfold = StratifiedKFold(n_splits=5, shuffle=False)
for train, valid in kfold.split(train_x, train_y):
    X_train = train_x[train]
    X_valid = train_x[valid]
    Y_train = train_y[train]
    Y_valid = train_y[valid]
    print(X_train.shape, Y_train.shape)
    model = CatBoostClassifier(
        iterations=1000,
        od_type='Iter',
        od_wait=120,
        # eval_metric = 'Accuracy',
        # verbose=50,
        max_depth=8,
        learning_rate=0.02,
        l2_leaf_reg=9,
        random_seed=2019,
        metric_period=50,
        fold_len_multiplier=1.1,
        loss_function='MultiClass',
        logging_level='Verbose'
    )
    model.fit(X_train,
              Y_train, eval_set=(X_valid, Y_valid), use_best_model=True
              )
    valid_pred = model.predict_proba(X_valid)
    print(valid_pred)
    valid_pred = valid_pred.argmax(axis=1)
    print(classification_report(Y_valid, valid_pred))
    results += model.predict_proba(test_x)
    del model
    gc.collect()
sub=pd.DataFrame()
pred=results.argmax(axis=1)
dd={0:1,1:4,2:5,3:21}
def sub_process(x):
    x=dd[x]
    return x
sub['label']=list(pred)
sub['label']=sub['label'].apply(sub_process)
sub['id']=list(test_id)
sub.to_csv('./result/baseline.csv',index=False)