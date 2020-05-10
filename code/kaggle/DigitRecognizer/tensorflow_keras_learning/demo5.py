#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo2
date: 2018/4/2
bref:
"""

import tensorflow as tf
import numpy as np


# 添加一层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1  # 默认不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 第1层，隐藏层10个单元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 第2层
predition = add_layer(l1, 10, 1, activation_function=None)

# 求损失和，
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 0.1 学习率

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(30000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 1000 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
