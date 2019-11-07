"""
Author: Zhou Chen
Date: 2019/11/7
Desc: 绘制相关图片
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoid():
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    plt.plot(x, sigmoid(x), c='b')
    plt.title("sigmoid function")
    plt.savefig('../assets/sigmoid.png')
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()