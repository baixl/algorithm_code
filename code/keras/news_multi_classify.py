# /us/bin/python3
# -*- coding:utf-8 -*-
# @Author:baixiaoling
# @Time:2018/12/02 
# comments: tensorflow教程， 基本分类
# python深度学习教程 新闻多分类
from keras.datasets import reuters
# 1、加载数据集
# 路透社1986年数据，共46个分类主题
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data), len(test_data))
print(train_data[0], train_labels[0])

word_index = reuters.get_word_index()
reverse_word_index = dict([(key, val) for (key, val) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
#2 准备数据，对数据编码
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# TODO: 自定义的one_hot函数, 也可以使用keras 内置的one_hot函数 to_categorical 
def to_one_hot(labels, dimension = 46):
    results = np.zeros(len(labels), dimension)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)
#  使用keras内置函数对标签数据one-hot处理
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# 3 构建网络
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax')) # 多分类
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 分割数据
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print('history_keys:\t',history_dict.keys())

# 6 绘制训练损失和验证损失
import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs =range(1, len(loss_values) +1)

plt.plot(epochs, loss_values, 'bo', label='train loss')
plt.plot(epochs, val_loss_values, 'bx',label = 'val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# 7 绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs, acc, 'ro', label='train acc')
plt.plot(epochs, val_acc, 'rx',label = 'val acc')
plt.title('train and val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

# 大概在第9轮开始过拟合
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print('results:\t', results)



# # 由损失和精度的图像可以看到，model过拟合了，大概在4epoches时效果最优
# # 重新训练
# model.fit(x_train, y_train, epochs=4, batch_size=512)
# result= model.evaluate(x_test, y_test)
# print(result)

# # 预测数据
# print(model.predict(x_test))
