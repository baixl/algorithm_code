#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: cut_words
date: 2017/11/29
bref: jieba 分词处理
"""
import os
import sys
import jieba
import jieba.analyse
import jieba.posseg as pseg


def cut_words(sentence):
    """
    切词
    :param sentence:
    :return:
    """
    return " ".join(jieba.cut(sentence)).encode('utf-8')


wiki_zh_path = sys.path['0'] + "/wiki-zh-1.3g.txt"

cut_words_out_path = sys.path['0'] + "/wiki-zh-1.3g_cut_words.txt"

count = 0
with open(wiki_zh_path, 'r') as f:

    for line in f:
        count += 1
        # if count % 10 == 0:
        if count ==10:
            print count
            break
        cut_words_arr = cut_words(line)
        print cut_words_arr
#
# target = open("wiki.zh.text.jian.seg", 'a+')
# print 'open files'
# line = f.readlines(100000)
# while line:
#     curr = []
#     for oneline in line:
#         # print(oneline)
#         curr.append(oneline)
#     '''
#     seg_list = jieba.cut_for_search(s)
#     words = pseg.cut(s)
#     for word, flag in words:
#         if flag != 'x':
#             print(word)
#     for x, w in jieba.analyse.extract_tags(s, withWeight=True):
#         print('%s %s' % (x, w))
#     '''
#     after_cut = map(cut_words, curr)
#     # print lin,
#     # for words in after_cut:
#     # print words
#     target.writelines(after_cut)
#     print 'saved 100000 articles'
#     line = f.readlines(100000)
# f.close()
# target.close()
