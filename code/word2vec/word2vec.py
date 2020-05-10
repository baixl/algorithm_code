#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: word2vec
date: 2017/11/27
bref:
"""

from gensim.models import word2vec

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
raw_sentences = ['the quick brown fox jumps over the lazy dogs ', 'yoyoyo you go home now to sleep']
sentences = [s.split() for s in raw_sentences]
# min_count 过滤掉小于等于min_count的词语，通常取0~100之前的数
# size 词向量的维数，默认值为100，通常取100到数百
word2vec_model = word2vec.Word2Vec(sentences, min_count=1, size=100)
print word2vec_model.similarity('fox', 'dogs')
print word2vec_model['dogs']