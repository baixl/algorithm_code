# /us/bin/python3
# -*- coding:utf-8 -*-
# @Author:baixiaoling
# @Time:2018/12/02 
# comments: tensorflow教程， 基本分类
# python深度学习教程 imdb:电影评分数据 二分类
from keras.datasets import imdb

# 1、加载数据集
# imdb 50000条评论，只保留数据中top10000个最常出现的
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 
print(train_data[0], train_labels[0])
print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
reverser_word_index = dict([(key, val) for (key, val) in word_index.items()])
decoded_review =  ' '.join([reverser_word_index.get(i-3, '?') for i in train_data[0]])

#2 准备数据，对数据编码
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train =  vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print('x_train_length:', len(x_train), '\tx_test_length:', len(x_test))

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 3 构建网络
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # 二分类，使用概率输出

# 4 编译模型
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
from keras import optimizers
from keras import  losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy,
              metrics=['accuracy'])

# 5 验证数据
# 分割训练集和测试集
x_val = x_train[1:10000]
partial_x_train = x_train[10000:]
y_val = y_train[1:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

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


# 由损失和精度的图像可以看到，model过拟合了，大概在4epoches时效果最优
# 重新训练
model.fit(x_train, y_train, epochs=4, batch_size=512)
result= model.evaluate(x_test, y_test)
print(result)

# 预测数据
print(model.predict(x_test))
