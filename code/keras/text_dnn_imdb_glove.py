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
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()


def load_data():
    """
    数据预处理
    """
    imdb_dir = 'aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith(".txt"):
                try:
                    f = open(os.path.join(dir_name, fname))
                    # print(os.path.join(dir_name, fname))
                    # print(f.read())
                    texts.append(f.read())
                    f.close()
                    if label_type == "neg":
                        labels.append(0)
                    else:
                        labels.append(1)
                except Exception as e:
                        pass
    return texts, labels


def data_handle(texts, labels, maxlen, traning_samples, validation_samples, max_words):
    """
    """
    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)
    print("data shape:", data.shape[0])
    print("label shape:", labels.shape[0])

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    print('data size:', len(data))

    x_train = data[:traning_samples]
    y_train = labels[:traning_samples]
    x_val = data[traning_samples:traning_samples + validation_samples]
    y_val = labels[traning_samples:traning_samples + validation_samples]

    print("train size[%d], val_sizes[%d]!" % (len(x_train), len(x_val)))
    return x_train, y_train, x_val, y_val, word_index


def load_glove(max_words, word_index):
    """
    """
    glove_dir = 'glove.6B'
    embedding_index = {}

    with open(os.path.join(glove_dir, "glove.6B.100d.txt"), 'rb') as f:
        for line in f:
            try:
                line_arr = line.split()
                word = line_arr[0]
                glove_vec = np.asarray(line_arr[1:], dtype='float32')
                embedding_index[word] = glove_vec
            except Exception as e:
                pass
    print("Found %s word vectors!" % len(embedding_index))

    embedding_dim = 100

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def model_func(max_words, embedding_dim, mnaxlen, embedding_matrix):
    """
    model部分
    """
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=mnaxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.layers[0].set_weights([embedding_matrix])  # 冻结嵌入层
    model.layers[0].trainable = False
    return model


def plot(history):
    """
    绘制学习曲线
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='train acc')
    plt.plot(epochs, val_acc, 'b', label='val acc')
    plt.title(' train and val accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    """
    # 1、加载数据
    texts, labels = load_data()

    maxlen = 100  # 评论超过100 截断
    traning_samples = 10000  # 训练集大小
    validation_samples = 2000  # 验证集大小
    max_words = 10000  # 只考虑前10000个常见单词
    embedding_dim = 100
    x_train, y_train, x_val, y_val, word_index = data_handle(
        texts, labels, maxlen, traning_samples, validation_samples, max_words)
    embedding_matrix = load_glove(max_words, word_index)

    model = model_func(max_words, embedding_dim, maxlen, embedding_matrix)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10,
                        batch_size=32, validation_data=(x_val, y_val))
    model.save_weights('pre_train_glove_mode.h5')

    plot(history)
