#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo2
date: 2018/4/2
bref:
"""

import tensorflow as tf

state =  tf.Variable(0, name='count')
one = tf.constant(1)


new_value = tf.add(state, one)
update = tf.assign(state,new_value)

init =  tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
