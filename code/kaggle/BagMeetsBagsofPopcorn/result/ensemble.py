#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: train
date: 2018/3/30
bref:逻辑
"""

path1 = "./gbdt_output.csv"
path2 = "./lr_output.csv"
path3 = "./MultinomialNB_prediction.csv"
path4 = "./nb_output.csv"
path5 = "./output_LR_chi2.csv"

temp = dict()

with open(path1, 'rb') as f:
    for line in f:

        list = line.decode("utf-8").strip("\n").split(",")
        if list[0] == "id": continue
        if list[0] in temp:
            temp[list[0]] += int(list[1])
        else:
            temp[list[0]] = int(list[1])

with open(path2, 'rb') as f:
    for line in f:
        list = line.decode("utf-8").strip("\n").split(",")
        if list[0] == "id": continue
        if list[0] in temp.keys():
            temp[list[0]] += int(list[1])
        else:
            temp[list[0]] = int(list[1])

with open(path3, 'rb') as f:
    for line in f:
        list = line.decode("utf-8").strip("\n").split(",")
        if list[0] == "id": continue
        if list[0] in temp.keys():
            temp[list[0]] += int(list[1])
        else:
            temp[list[0]] = int(list[1])
with open(path4, 'rb') as f:
    for line in f:
        list = line.decode("utf-8").strip("\n").split(",")
        if list[0] == "id": continue
        if list[0] in temp.keys():
            temp[list[0]] += int(list[1])
        else:
            temp[list[0]] = int(list[1])
with open(path5, 'rb') as f:
    for line in f:
        list = line.decode("utf-8").strip("\n").split(",")
        if list[0] == "id": continue
        if list[0] in temp.keys():
            temp[list[0]] += int(list[1])
        else:
            temp[list[0]] = int(list[1])
out_path = './out.csv'
with open(out_path, 'a+') as f:
    for id, value in temp.items():
        if value >= 3:
            f.write(id + ",1\n")
        else:
            f.write(id + ",0\n")
print(temp)
