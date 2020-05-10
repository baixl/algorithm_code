#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo
date: 2018/4/2
bref:
"""
import tensorflow as tf
import pandas as pd
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -0.1, 1.0))  # 一维结构，范围 -1 到1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5:学习率
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

sess =  tf.Session()
sess.run(init)  # 激活init

for step in range(500): #训练200步
    sess.run(train) # 训练
    if step %20 ==0:
        print(step, sess.run(Weights), sess.run(biases))
