#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: tensorflow_softmax
date: 2018/4/2
bref: 使用cnn结构实现 能够达到
"""
import numpy as np
import pandas as pd
import tensorflow as tf

# 1 数据处理
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_input = train_data.iloc[:, 1:].values.astype(np.float)
train_input = np.multiply(train_input, 1.0 / 255.0)
train_label = train_data.iloc[:, 0].values.astype(np.float)

test_input = test_data.values.astype(np.float)
test_input = np.multiply(test_input, 1.0 / 255.0)

IMAGE_SIZE = len(train_input[0])
print("image_size : %d" % (IMAGE_SIZE))
IMAGE_CLASSES = 10
VALIDATION_SPLIT = 0.2


def onehot_encoder(label, classes):
    label_size = label.shape[0]
    label_onehot = np.zeros((label_size, classes))
    for i in range(label_size):
        label_onehot[i][int(label[i])] = 1
    return label_onehot


train_label = onehot_encoder(train_label, IMAGE_CLASSES)


def split_train_valid(train_input, train_label):
    indices = np.arange(train_input.shape[0])
    np.random.shuffle(indices)
    train_input = train_input[indices]
    train_label = train_label[indices]
    validation_samples = int(VALIDATION_SPLIT * train_input.shape[0])

    X_train = train_input[:-validation_samples]
    Y_train = train_label[:-validation_samples]
    X_val = train_input[-validation_samples:]
    Y_val = train_label[-validation_samples:]

    return X_train, Y_train, X_val, Y_val


epoch = 4
batch_size = 50
max_steps = 1000
index_in_epoch = 0
learning_rate = 1e-3


# get training batchs
def next_batch(X_train, Y_train, batch_size):
    global index_in_epoch

    if index_in_epoch + batch_size <= X_train.shape[0]:
        X_train_batch = X_train[index_in_epoch: index_in_epoch + batch_size]
        Y_train_batch = Y_train[index_in_epoch: index_in_epoch + batch_size]

        index_in_epoch += batch_size

    else:
        index_in_epoch = 0

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        X_train_batch = X_train[index_in_epoch: index_in_epoch + batch_size]
        Y_train_batch = Y_train[index_in_epoch: index_in_epoch + batch_size]

        index_in_epoch += batch_size

    return X_train, Y_train, X_train_batch, Y_train_batch


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784])  # 28x28
    y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 10个值
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 32], dtype=tf.float32, stddev=1e-1, name='weight'))
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32), name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name='conv1')

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')

    with tf.name_scope('conv2'):
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype=tf.float32, stddev=1e-1, name='weight'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='biases')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases, ), name='conv2')

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool2')



    dropout1 =  tf.nn.dropout(pool2, keep_prob)

    dropout_flat = tf.reshape(pool2, [-1, 7*7*64])


    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], dtype=tf.float32, stddev=1e-1), name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), name='b_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(dropout_flat, W_fc1) + b_fc1)

    dropout3 = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-1), name='W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), name='b_fc2')
    y_conv = tf.nn.softmax(tf.matmul(dropout3, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])

    # 注意这里的优化器是AdamOptimizer， 不是AdadeltaOptimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    predict = tf.argmax(y_conv, 1)
    # 5 训练
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        val_accuracy = []

        for i in range(epoch):
            X_train, Y_train, X_val, Y_val = split_train_valid(train_input, train_label)
            print("training epoch %d" % (i + 1))

            for step in range(max_steps):
                X_train, Y_train, X_batch, Y_batch = next_batch(X_train, Y_train, batch_size)
                sess.run(train_step, feed_dict={
                    x: X_batch,
                    y_: Y_batch,
                    keep_prob: 0.5
                })
                if step % 100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x: X_batch,
                        y_: Y_batch,
                        keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (step, train_accuracy))
            accuracy_ = sess.run(accuracy, feed_dict={x: X_val, y_: Y_val, keep_prob: 1.0})
            val_accuracy.append(accuracy_)

        val_accuracy = np.array(val_accuracy)
        val_accuracy_mean = val_accuracy.mean()
        print("test accuracy %g" % val_accuracy_mean)

        predicts = np.zeros(test_input.shape[0])
        for i in range(0, test_input.shape[0] // batch_size):
            predicts[i * batch_size: i * batch_size + batch_size] = sess.run(predict,
                                                                             feed_dict={
                                                                                 x: test_input[
                                                                                    i * batch_size: i * batch_size + batch_size],
                                                                                 keep_prob: 1.0})

        submissions = pd.DataFrame({'ImageId': np.arange(1, 1 + test_input.shape[0]), 'Label': predicts.astype(int)})
        submissions.to_csv('../result/submission_cnn1.csv', index=False)
