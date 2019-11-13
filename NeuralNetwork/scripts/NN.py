"""
Author: Zhou Chen
Date: 2019/11/13
Desc: About
"""
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sigmoid_grad(x):
    return x * (1-x)


def one_hot(x):
    array = np.zeros(shape=[10, 1])
    array[x-1, 0] = 1
    return array


def plot_his(his):
    """
    绘制训练过程
    :param his:
    :return:
    """
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='loss')
    plt.plot(np.arange(len(his['accuracy'])), his['accuracy'], label='accuracy')
    plt.title('training history')
    plt.legend(loc=0)
    plt.show()


def mse(y_label, y_pred):
    y_pred = np.squeeze(y_pred, axis=-1)
    if y_label.shape == y_pred.shape:
        return np.sum((y - y_pred)**2 / y.shape[0])
    else:
        print("no match shape")
        return None


class BPNet(object):

    def __init__(self):
        """
        构建单隐层神经网络
        """
        self.weights = None  # 列表存放两层的参数
        self.bias = None
        self.history = {'loss': [], 'accuracy': []}  # loss和accuracy的历史

    def train(self, x, y, trained_weights=None, learning_rate=1e-3, epochs=100):
        if trained_weights:
            # 传入预训练参数进行训练
            self.weights = [trained_weights[0][:, 1:], trained_weights[1][:, 1:]]
            self.bias = [trained_weights[0][:, 0], trained_weights[1][:, 0]]
        else:
            # 未传入则初始化参数
            print("init weights")
            self.weights = [np.random.normal(size=[25, 400]), np.random.normal(size=[10, 25])]
            self.bias = [np.random.normal(size=[25, 1]), np.random.normal(size=[10, 1])]

        for epoch in range(epochs):
            for i in range(x.shape[0]):
                img = x[i].reshape(-1, 1)
                label = y[i].reshape(-1, 1)
                # forward
                # 隐藏层计算
                input_hidden = self.weights[0] @ img + self.bias[0].reshape(-1, 1)   # [25, 400] @ [400, 1] + [25, 1] => [25, 1]
                output_hidden = sigmoid(input_hidden)
                # 输出层计算
                input_output = self.weights[1] @ output_hidden + self.bias[1].reshape(-1, 1)   # [10, 25] @ [25, 1] + [10, 1] => [10, 1]
                output_output = sigmoid(input_output)
                # backward
                output_error = sigmoid_grad(output_output) * (label - output_output)
                hidden_error = sigmoid_grad(output_hidden) * (self.weights[1].T @ output_error)
                # 参数更新
                self.weights[1] += (output_error @ output_hidden.T) * learning_rate
                self.bias[1] += output_error * learning_rate
                self.weights[0] += (hidden_error @ img.T) * learning_rate
                self.bias[0] += hidden_error * learning_rate
            # 计算损失
            pred_epoch = np.argmax(np.squeeze(self.predict(x), axis=-1), axis=1)
            y_true = np.argmax(y, axis=1)  # onehot还原为标签
            acc = np.sum(pred_epoch.reshape(-1, 1) == y_true.reshape(-1, 1)) / y.shape[0]
            loss = mse(y, self.predict(x))
            self.history['loss'].append(loss)
            self.history['accuracy'].append(acc)
            print("epoch {}, loss {}, accuracy {}".format(epoch, loss, acc))

            # 添加早停
            if epoch > 10 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < 1e-5:
                break
        return self.history

    def predict(self, x, trained_weights=None):
        if trained_weights:
            # 传入预训练参数进行预测
            self.weights = [trained_weights[0][:, 1:], trained_weights[1][:, 1:]]
            self.bias = [trained_weights[0][:, 0], trained_weights[1][:, 0]]

        if self.weights is None:
            print("no weights, cannot predict")

        result = []
        for i in range(x.shape[0]):
            img = x[i].reshape(-1, 1)
            # 隐藏层计算
            input_hidden = self.weights[0] @ img + self.bias[0].reshape(-1, 1)  # [25, 400] @ [400, 1] + [25, 1] => [25, 1]
            output_hidden = sigmoid(input_hidden)
            # 输出层计算
            input_output = self.weights[1] @ output_hidden + self.bias[1].reshape(-1, 1)  # [10, 25] @ [25, 1] + [10, 1] => [10, 1]
            output_output = sigmoid(input_output)
            result.append(output_output)
        return np.array(result)


if __name__ == '__main__':
    data = scio.loadmat('../data/ex3data1.mat')  # 读取数据集
    pretrained_weights = scio.loadmat('../data/ex3weights.mat')  # 读取预训练参数
    X, y = data['X'], data['y']  # 按照索引取出data和label
    y = LabelBinarizer().fit_transform(y)
    w_hidden, w_output = pretrained_weights['Theta1'], pretrained_weights['Theta2']
    # 构建模型
    net = BPNet()
    # 载入预训练参数
    pred_result = net.predict(X, [w_hidden, w_output])
    pred_result = np.argmax(np.squeeze(pred_result, axis=-1), axis=1)
    y_true = np.argmax(y, axis=1)
    print("载入参数前向传播准确率", np.sum(pred_result.reshape(-1, 1) == y_true.reshape(-1, 1)) / y.shape[0])
    # 利用BP算法训练网络模型
    his = net.train(X, y, learning_rate=1e-1, epochs=200)
    plot_his(his)
