# 机器学习实战练习

随书下载地址：
```
http://www.ituring.com.cn/book/1021
http://www.ituring.com.cn/book/download/0019ab9d-0fda-4c17-941b-afe639fcccac
```

基于《机器学习实战》在代码中联系ml算法

## chapter2: knn 
代码：knn.py  
数据：  
testDigits  
trainingDigits  
datingTestSet.txt  
datingTestSet2.txt  

两个例子： 
* 约会网站数据分类(讲了数据的归一化处理)
* 基于knn实现手写数字识别，书中demo错误率1.2 

>我的代码中应该是哪里出错了，错误率一直很高

## chapter3: decision tree
* ID3算法，使用信息增益构建决策树
以一个特征进行划分：选择是信息熵下降最多的特征作为树分叉的依据

* 信息熵：描绘了数据包含的信息量大小，分类也准确，包含的信息也就越少。
数据越分散（不确定）包含的信息越多，对应熵越高。

可以使用matplotlib的注解工具 annotations 画出决策树的图形
## chapter4: naive bayesian

* 贝叶斯准则
p(c|x) = p(x|c)p(c)/p(x)
* 条件独立性假设
## chapter 5 logistic 回归
logistic函数：  
logistic(x) = 1/(1+exp(-z))

z= x0 + w1x2+w2x2+....+ wnxn = WT*x

参数优化求解算法：

梯度上升/下降

当数据量很大时，使用随机梯度上升/下降

* 数据缺失值的处理
    * 使用可用特征的均值填补缺失值
    * 使用特殊值来填补缺失值 如-1
    * 忽略缺失样本
    * 使用相似样本均值填补缺失值
    * 使用另外的机器学习算法预测缺失值
    
    




