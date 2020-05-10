# coding:utf-8

import torch.optim as optim
from torch.nn import init
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(
    0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)

# 使用pytorch提供的工具读取数据

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break

# 定义模型 使用nn.Module定义模型


class LinerNet(nn.Module):
    def __init__(self, n_features):
        super(LinerNet, self).__init__()
        self.liner = nn.Linear(n_features, 1)  # input_size, output_size

    def forward(self, x):
        y = self.liner(x)
        return y


net = LinerNet(num_inputs)
print(net)  # Linear(in_features=2, out_features=1, bias=True)
# 或者使用Sequential
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print(net)

for param in net.parameters():
    print(param)

# 初始化参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
# 定义损失函数
loss = nn.MSELoss()
# 定义优化方法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d loss %f' % (epoch, l.item()))

dense = net[0]  # 访问第0层
print(true_w, dense.weight)
print(true_b, dense.bias)
