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
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.005, (200,))
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

for step in range(3001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print("step:%d,   training cost:%f" % (step, cost))
print("test")

cost = model.evaluate(X_test, Y_test, batch_size=40)

print("test cost:", cost)

W, b = model.layers[0].get_weights()

print("weights:", W, "\tbiases:", b)

print("打印预测")
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
