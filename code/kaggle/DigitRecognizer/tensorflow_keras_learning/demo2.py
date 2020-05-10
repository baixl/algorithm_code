#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo2
date: 2018/4/2
bref:
"""

import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

#不需要自己手动关闭
with tf.Session() as sees:
    result2  = sees.run(product)
    print(result2)