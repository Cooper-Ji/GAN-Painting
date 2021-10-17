
from abc import ABC
import torch
import torch.nn.functional as F  # 激励函数都在这
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.net = nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    # x = torch.zeros
    # y = x + 100
    # print(x)

    # n_data = torch.ones(100, 1)
    # x = torch.normal(-2 * n_data, 1)
    # y = x + 100
    # print(x)

    # 用 Variable 来修饰这些数据 tensor
    x, y = torch.autograd.Variable(x), Variable(y)

    # 画图
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    net = Net(n_feature=1, n_hidden=10, n_output=1)
    # optimizer 是训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

    plt.ion()  # 画图
    plt.show()

    for t in range(200):
        prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值
        # for i in range(100):
        #     print(x.data.numpy()[i][0], prediction.data.numpy()[i][0], y.data.numpy()[i][0])

        loss = loss_func(prediction, y)  # 计算两者的误差
        # print(loss.data.numpy())
        # print("\n\n\n")

        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

        # 接着上面来
        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.5)
        # print(t)

