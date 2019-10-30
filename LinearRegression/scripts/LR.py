"""
Author: Zhou Chen
Date: 2019/10/30
Desc: Linear Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
fig = plt.gcf()
fig.set_size_inches(12, 6)

data = pd.read_csv("../data/data.csv")
x = data['square_feet'].values
y = data['price'].values


def plot_h(x, y, theta, loss):
    plt.scatter(x[:, 1].reshape(-1, 1), y, color='b')
    plt.plot(x[:, 1].reshape(-1, 1), x@theta, color='r', linewidth=1, marker='o')
    plt.text(500, 10000, 'Loss={:.2f}'.format(loss), fontdict={'size': 15, 'color': 'red'})


def loss_func(x, y, theta):
    pred = x @ theta
    loss = np.sum((pred - y.reshape(-1, 1)) ** 2) / (x.shape[0]*2)
    return loss


class LinearRegression(object):
    """
    实现单变量线性回归模型
    """

    def __init__(self):
        self.theta = np.zeros([2, 1])

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=10):
        """
        实现回归模型的训练
        :param x_train:
        :param y_train:
        :return:
        """
        plt.ion()
        for epoch in range(epochs):
            x = np.hstack((np.ones([x_train.shape[0], 1]), x_train.reshape(-1, 1)))
            grad = (x @ self.theta) - y_train.reshape(-1, 1)
            self.theta[0, 0] = self.theta[0, 0] - learning_rate * (np.sum(grad) / x.shape[0])
            self.theta[1, 0] = self.theta[1, 0] - learning_rate * (np.sum((grad * x[:, 1].reshape(-1, 1))) / x.shape[0])
            loss = loss_func(x, y_train, self.theta)
            print('loss', loss)
            import time
            time.sleep(1)
            plt.cla()
            plot_h(x, y_train, self.theta, loss)

            plt.pause(0.1)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    lr = LinearRegression()
    lr.fit(x, y, learning_rate=0.000001, epochs=30)