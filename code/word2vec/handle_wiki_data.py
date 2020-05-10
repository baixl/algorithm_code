#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xiaolingbai@gmail.com
file: demo
date: 2017/10/26
bref: 解析wiki中文1.3g文件，
1 抽取文本
2 转化成简体中文  参考 http://blog.csdn.net/thomashtq/article/details/41512359

这个文件不做分词处理：这里只是获得wiki百科的原始中文数据
后续jieba分词（是否去除停用词等）在后续处理
"""

import os
import time
from gensim.corpora import WikiCorpus
from langconv import *


import multiprocessing


# 简体繁体的转换
def simple2tradition(line):
    # 将简体转换成繁体
    line = Converter('zh-hant').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line


def tradition2simple(line):
    # 将繁体转换成简体
    line = Converter('zh-hans').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line


def handle_wiki_data():
    print os.getcwd()
    wiki_source_path = sys.path[0] + '/zhwiki-latest-pages-articles.xml.bz2'
    print 'wiki baike 数据路径', wiki_source_path
    # 解析后终保存的目录
    wiki_source_path_out = sys.path[0] + '/wiki-zh-1.3g.txt'
    print 'wiki baike 数据解析结果路径', wiki_source_path_out
    wiki = WikiCorpus(wiki_source_path, lemmatize=False, dictionary={})
    file_out = open(wiki_source_path_out, 'w')
    for text in wiki.get_texts():
        str_line = ' '.join(text) + "\n"
        # 转化成简体中文
        simple_line = tradition2simple(str_line.encode('utf-8'))
        file_out.write(simple_line)
    file_out.close()


# class Producer(multiprocessing.Process):
#     """
#     生产者：读取wiki百科数据，放入队列，让生产者处理
#     """
#
#     def __init__(self, queue, end_flag):
#         """
#         init function
#         """
#         multiprocessing.Process.__init__(self)
#         self.queue = queue
#         self.end_flag = end_flag
#
#     def run(self):
#         wiki_source_path = sys.path[0] + '/zhwiki-latest-pages-articles.xml.bz2'
#         print 'wiki baike 数据路径', wiki_source_path
#         # 解析后终保存的目录
#         wiki = WikiCorpus(wiki_source_path, lemmatize=False, dictionary={})
#         # count = 0
#         for text in wiki.get_texts():
#             # count += 1
#             # if count == 1000:
#             #     break
#             self.queue.put(text)
#         self.queue.put(self.end_flag)
#
#
# class Consumer(multiprocessing.Process):
#     def __init__(self, name, queue, end_flag, file_out):
#         """
#         init function
#         """
#         multiprocessing.Process.__init__(self)
#         self.name = name
#         self.queue = queue
#         self.end_flag = end_flag
#         self.file_out = file_out
#
#     def run(self):
#         print "进程: " + self.name
#         while True:
#             text = self.queue.get()
#             if text == self.end_flag:
#                 self.queue.put(self.end_flag)
#                 self.file_out.close()
#                 break
#             else:
#                 str_line = ' '.join(text) + "\n"
#                 # 转化成简体中文
#                 simple_line = tradition2simple(str_line.encode('utf-8'))
#                 self.file_out.write(simple_line)


if __name__ == '__main__':
    start_time = time.time()
    # wiki_source_path_out = sys.path[0] + '/wiki-zh-1.3g.txt'
    # end_flag = False
    # que = multiprocessing.Queue()
    # # 进程数量
    count = multiprocessing.cpu_count()-1
    count = 2
    # consumer = []
    # for i in range(count):
    #     file = open(wiki_source_path_out + '_' + str(i), 'w')
    #     consumer.append(Consumer('consumer_' + str(i), que, end_flag, file))
    #     print '初始化 consumer_' + str(i)
    # p = Producer(que, end_flag)
    # p.start()
    # for c in consumer:
    #     c.start()
    # p.join()
    # for c in consumer:
    #     c.join()
    # end_time = time.time()
    # print "总过处理时间:", end_time - start_time

    handle_wiki_data()
    print '单线程时间', time.time() - end_time
