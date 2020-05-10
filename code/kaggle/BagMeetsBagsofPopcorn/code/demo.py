#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: demo
date: 2018/3/30
bref:
"""
path = "/Users/baixiaoling/Desktop/ensemble_final.csv"
file =open(path+"out", 'a+')
with open(path, 'rb') as f:
    for line in f:
        list = line.decode("utf-8").strip("\n").split(",")
        result =0
        if float(list[1]) > 0.5:
            result = 1
        list = [list[0], str(result)]
        print(" ".join(list))
        file.write(",".join(list)+"\n")

file.close()

