#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: train_word2vec
date: 2018/3/30
bref: 训练word2vec
"""

from  gensim.models import word2vec
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords




raw_sentences = ['the quick brown fox jumps over the lazy dogs ', 'yoyoyo you go home now to sleep']
sentences = [s.split() for s in raw_sentences]

print(sentences)

# train = pd.read_csv("../input/unlabeledTrainData.tsv", delimiter='\t')

# print(len(train['review']))


path ="../input/clean_unlabeledTrainData.tsv"
sentences = []

with open(path, "rb") as f:
    for line in f:
        sentences.append(str(line, encoding="utf-8").split("\t")[1].split(" "))


# print(sentences)

from gensim.models import Word2Vec

# 模型参数
num_features = 300  # Word vector dimensionality
min_word_count = 20  # Minimum word count
num_workers = 5  # Number of threads to run in parallel
context = 20  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
                 size=num_features, min_count=min_word_count, \
                 window=context, sample=downsampling)

# 训练模型中...
# CPU times: user 6min 16s, sys: 8.34 s, total: 6min 24s
# Wall time: 2min 27s
print('保存模型...')
model.init_sims(replace=True)
model_name = "./300features_20minwords_20context"
model.save(model_name)
