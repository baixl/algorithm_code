#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: tensorflow_softmax
date: 2018/4/2
bref: 使用cnn结构实现
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


# 输入和输出
x = tf.placeholder(tf.float32, shape=[None, 784])  # 28x28
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 10个值
keep_prob = tf.placeholder(tf.float32)


# 2 tensorflow 权重、cnn、pool函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 初始化为大于0的一个很小的数
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 3 cnn结构   卷积1+ max pool+卷积2+max pool +卷积3+max pool +  full connect + dropout + full connect
# 3.1 卷基层1+pool层1
# conv1 layer  window：5*5  32个filter
W_conv1 = weight_variable([3, 3, 1, 36])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([36])

# (40000,784) => (40000,28,28,1)
image = tf.reshape(x, [-1, 28, 28, 1])

# 激活函数
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)

# 3.2 卷基层2+pool层2

# conv2 layer
W_conv2 = weight_variable([3, 3, 36, 36])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # (40000, 7, 7, 64)

## 3.3
# conv3 layer
W_conv3 = weight_variable([3, 3, 36, 36])  # patch 5x5, in size 32, out size 64
b_conv3 = bias_variable([36])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # output size 14x14x64
h_pool3 = max_pool_2x2(h_conv3)  # (40000, 7, 7, 64)

# 3.4全连接层1
# fullc1 layer
W_fc1 = weight_variable([4 * 4 * 36, 576])
b_fc1 = bias_variable([576])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 36])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# 3.5 dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 3.6 全连接层2(输出）
## fullc2 layer ##
W_fc2 = weight_variable([576, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # loss

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])


# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))#

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 4 prediction
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
                keep_prob: 0.33
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
                                                                             x: test_input[i * batch_size: i * batch_size + batch_size],
                                                                             keep_prob: 1.0})

    submissions = pd.DataFrame({'ImageId': np.arange(1, 1 + test_input.shape[0]), 'Label': predicts.astype(int)})
    submissions.to_csv('../result/submission_cnn.csv', index=False)
    #
    # np.savetxt('submission_cnn.csv',
    #        np.c_[range(1, len(test_images) + 1), predicted_lables],
    #        delimiter=',',
    #        header='ImageId,Label',
    #        comments='',
    #        fmt='%d')
