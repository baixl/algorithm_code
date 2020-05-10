# /us/bin/python3
# -*- coding:utf-8 -*-
# @FileName: baisc_classify.py
# @Author:baixiaoling
# @Time:2018/11/25 10:42
# comments: tensorflow教程， 基本分类
# https://tensorflow.google.cn/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 数据集：fashion mnist： 70000张服装图片，60000张用于训练，10000张用于评估

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 这里会下载数据，可以先将数据下载下来，然后放到  C:\Users\baixiaoling\.keras\datasets\fashion-mnist 下
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape, len(train_images))
print(train_labels, len(train_labels))

# ## 数据预处理
# plt.figure()
# plt.imshow(train_images[10])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# # 数值缩放
#
train_images, test_images = train_images / 255.0, test_images / 255.0
# #显示前25张图像
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 构建模型

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Flatten, 将图像从二维28 x 28 转化成一维 28 * 28 = 784
# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=5)
# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test accuracy:', test_acc)
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax((predictions[0])))
print(test_labels[0])
