# /us/bin/python3
# -*- coding:utf-8 -*-
# @Author:baixiaoling
# @Time:2018/12/02 
# comments: tensorflow教程， 基本分类
# python深度学习教程 波斯顿房价预测， 使用交叉验证
from keras.datasets import boston_housing

# 1、加载数据集
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(len(train_data), len(test_data))
print(train_data.shape, test_data.shape)
# 2 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# 注意：测试数据的标准化也是使用训练数据的方差和标准差，不能使用测试数据的任何计算值
test_data -= mean
test_data /= std

# 3 构建网络

from keras import layers
from keras import models


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # mse:均方误差 mae：平均绝对误差
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 4、使用k折交叉验证
import numpy as np

k = 4
num_val_samples = len(train_data) // k  # // 除法运算，返回商的整数部分
num_epochs = 200
all_scores = []
all_mae_historiews = []
for i in range(k):
    print('processing fold#:\t', i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    # verbose =0  训练模式，静默模式
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_historiews.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_historiews]) for i in range(num_epochs)]

# # 5 绘制验证分数

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('epochs')
plt.ylabel('validation mae')
plt.show()


# 使用滑动平均模型，平滑这个函数
def smooth_curve(points, factor=0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            prev = smooth_points[-1]
            smooth_points.append(prev * factor + point * (1 - factor))
        else:
            smooth_points.append(point)
    return smooth_points
smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('epochs')
plt.ylabel('validation mae')
plt.show()