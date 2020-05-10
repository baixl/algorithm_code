#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xlingbai@gmail.com
file: svd
date: 2018/1/21
bref:
"""
import numpy as np

U, Sigma, VT = np.linalg.svd([[1, 1], [7, 7]])

print U
print Sigma
print VT
