# /us/bin/python3
# -*- coding:utf-8 -*-
# @Author:baixiaoling
# @Time:2018/12/02
# comments: 使用glove词嵌入的文本分类
# imdb原始数据：acllmdb： 下载地址http://mng.bz/0tIo
# glove词典下载地址：https://nlp.stanford.edu/projects/glove/  glove.6B.zip

import os

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Embedding,SimpleRNN
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense

max_features = 10000
max_len = 500
batch_size = 32
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
