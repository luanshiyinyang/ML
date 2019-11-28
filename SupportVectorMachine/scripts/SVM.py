"""
Author: Zhou Chen
Date: 2019/11/28
Desc: SVM Demo
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def build_model():
    """
    建立模型
    :return:
    """
    svc = SVC(C=1.0, kernel='rbf', gamma='scale')
    return svc


if __name__ == '__main__':
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
    model = build_model()
    model.fit(x_train, y_train)
    print("准确率: {:.4f}".format(np.sum(y_test == model.predict(x_test)) / y_test.shape[0]))

    # 交叉验证效果
    kf = KFold(n_splits=5)
    i = 0
    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        test_x, test_y = x[test_index], y[test_index]
        model = build_model()
        model.fit(train_x, train_y)
        acc = np.sum(test_y == model.predict(test_x)) / test_y.shape[0]
        print("Fold {}, test accuracy {:.2f}".format(i+1, acc))
        i += 1