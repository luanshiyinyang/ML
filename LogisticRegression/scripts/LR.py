"""
Author: Zhou Chen
Date: 2019/11/8
Desc: 逻辑回归实现分类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('fivethirtyeight')


def plot_data(x, y, labels, theta):
    labels = labels.astype(np.int)
    cmap = cm.rainbow(np.linspace(0.0, 1.0, 2))
    colors = cmap[labels.reshape(-1)]
    plt.scatter(x, y, c=colors)
    theta_0 = theta[0, 0]
    theta_1 = theta[1, 0]
    theta_2 = theta[2, 0]
    hori_x = np.linspace(min(x), max(x), 100)
    ver_y = (- theta_1 * hori_x - theta_0) / theta_2
    plt.plot(hori_x, ver_y, c='g')
    plt.title('data')
    # plt.xlim([min(x)-10, max(x)+10])
    # plt.ylim([min(y)-10, max(y)+10])
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def CELoss(y_true, y_pred):
    return np.sum((-y_true * np.log(y_pred)) - ((1-y_true) * np.log(1-y_pred))) / len(y_true)


class LogisticRegression(object):
    """
    逻辑回归模型实现
    """

    def __init__(self):

        # self.theta = np.zeros([3, 1])
        self.theta = np.array([-24, 0.2, 0.2]).reshape(-1, 1)

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=10):
        """
        实现回归模型的训练
        :param x_train:
        :param y_train:
        :param learning_rate:
        :param epochs:
        :return:
        """
        plt.ion()
        for epoch in range(epochs):

            x = np.hstack((np.ones([x_train.shape[0], 1]), x_train))
            loss = CELoss(y_train, sigmoid(x @ self.theta))
            print('epoch', epoch, 'loss', loss)
            grad = (sigmoid(x @ self.theta)) - y_train.reshape(-1, 1)
            self.theta[0, 0] = self.theta[0, 0] - learning_rate * (np.sum((grad * x[:, 0].reshape(-1, 1))) / x.shape[0])
            self.theta[1, 0] = self.theta[1, 0] - learning_rate * (np.sum((grad * x[:, 1].reshape(-1, 1))) / x.shape[0])
            self.theta[2, 0] = self.theta[2, 0] - learning_rate * (np.sum((grad * x[:, 2].reshape(-1, 1))) / x.shape[0])
            loss = CELoss(y_train, sigmoid(x @ self.theta))

            plt.cla()
            plot_data(x[:, 1], x[:, 2], y_train, self.theta)

            plt.pause(0.1)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("../data/data.csv", encoding="utf8")
    x, y = data.values[:, :2], data.values[:, 2].reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(x, y, learning_rate=0.0001, epochs=1000)