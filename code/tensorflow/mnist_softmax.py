#!/usr/bin python
# *-*coding:utf-8-*-
"""
mnist：tensorflow官方教程
softmax回归模型，在测试集上的准确率：91%
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import datetime
start_time = datetime.datetime.now()

mnist = input_data.read_data_sets('./data/mnist_data', one_hot=True)
x = tf.placeholder("float", [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run((init))

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


end_time = datetime.datetime.now()
print((end_time - start_time).seconds)