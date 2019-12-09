"""
Author: Zhou Chen
Date: 2019/12/9
Desc: 简单实现Kmeans聚类算法
"""
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.datasets import load_iris
plt.style.use('fivethirtyeight')


def load_data():
    data, target = load_iris()['data'], load_iris()['target']
    # 选取前两列特征，方便可视化
    return data[:, :2], target


def plot_data(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.savefig('rst.png')
    plt.show()


def rand_centroids(data, k):
    m = data.shape[0]
    # 随机选择k个样本作为初始化的聚类中心
    sample_index = random.sample(list(range(m)), k)
    centroids = data[sample_index]
    # 循环遍历特征值
    return centroids


def compute_dist(vecA, vecB):
    return np.linalg.norm(vecA - vecB)


def kMeans(data, k):
    m, n = np.shape(data)  # 样本量和特征量
    labels = np.array(np.zeros((m, 2)))
    centroids = rand_centroids(data, k)
    cluster_changed = True  # 是否已经收敛
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                distJI = compute_dist(centroids[j, :], data[i, :])
                if distJI < min_dist:
                    min_dist = distJI
                    min_index = j
            if labels[i, 0] != min_index:
                cluster_changed = True
            labels[i, :] = min_index, min_dist ** 2
        for cent in range(k):
            ptsInClust = data[np.nonzero(labels[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果即类别
    return centroids, labels


if __name__ == '__main__':
    data, target = load_data()
    # plot_data(data, target)
    _, label = kMeans(data, 3)
    plot_data(data, label[:, 0])