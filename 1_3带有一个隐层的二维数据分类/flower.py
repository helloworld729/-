import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

#%matplotlib inline #如果你使用用的是Jupyter Notebook的话请取消注释。

np.random.seed(1) #设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。

X, Y = load_planar_dataset()  # 加载数据

# print('矩阵X维度：{}'.format(X.shape))
# print(X[:,0:6])
# print('矩阵Y维度：{}'.format(Y.shape))
# print(Y[:,0:11])


# plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), cmap=plt.cm.Spectral)
# plt.show()


#  逻辑斯蒂回归测试
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
#
# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
#         np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")


def layer_sizes(X, Y):
    """
    定义神经网络各个层节点的的数量
    :param X:
    :param Y:
    :return:
    """
    n_x = X.shape[0]  # 2
    n_h = 4
    n_y = Y.shape[0]  # 1

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化参数
    :param n_x: 输入层节点数
    :param n_h: 隐层节点数
    :param n_y: 输出层节点数
    :return: 参数字典
    """
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x)
    b1 = np.random.randn(n_h, 1)
    w2 = np.random.randn(n_y, n_h)
    b2 = np.random.randn(n_y, 1)


    parameter = {
        'w1': w1,
        'w2': w2,
        'b1': b1,
        'b2': b2,
    }

    return parameter


def forward_propagation(x, parameters):
    """
    前向传播
    :param x: 训练样本集
    :param parameters: 权重系数
    :return: 预测值a2，参数字典
    """
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) +b2
    a2 = sigmoid(z2)

    cache = {
        'z1': z1,
        'z2': z2,
        'a1': a1,
        'a2': a2
    }

    return a2, cache


def compute_cost(a2, y):
    """
    计算损失
    :param a2:也测结果向量
    :param y: 实际的label
    :return:  损失均值
    """
    m = y.shape[1]
    logprobs = np.multiply(np.log(a2), y) + np.multiply((1 - y), np.log(1 - a2))
    cost = -1 * (np.sum(logprobs)) / m
    cost = float(np.squeeze(cost))
    return cost


def backward_propagation(parameters, cache, x, y):
    """
    反向传播
    :param parameters: 权重参数
    :param cache:
    :param x: 样本集
    :param y: underground_truth
    :return: 偏导数
    """
    m = x.shape[1]
    a1 = cache['a1']
    a2 = cache['a2']
    w1 = parameters['w1']
    w2 = parameters['w2']
    dz2 = a2 - y
    dw2 = 1/m * np.dot(dz2, a1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = 1/m * np.dot(dz1, x.T)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        'dw1': dw1,
        'dw2': dw2,
        'db1': db1,
        'db2': db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    更新参数
    :param parameters: 上一轮的w b
    :param grads: 反向传播求得偏导数
    :param learning_rate: 学习率
    :return: 更新的参数
    """
    w1, w2 = parameters['w1'], parameters['w2']
    b1, b2 = parameters['b1'], parameters['b2']
    dw1, dw2 = grads['dw1'], grads['dw2']
    db1, db2 = grads['db1'], grads['db2']
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    paras = {
        'w1': w1,
        'w2': w2,
        'b1': b1,
        'b2': b2
    }

    return paras


def nn_model(X, Y, iterations, print_cost=True):
    n_x, n_h, n_y = layer_sizes(X,Y)  # 获取各个层的节点数目
    parameters = initialize_parameters(n_x, n_h, n_y)  # 初始权重

    for i in range(iterations):
        a2, cache = forward_propagation(X, parameters)  # 前向传播，获得预测值和z， a
        cost = compute_cost(a2, Y)  #
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, 1.2)

        if print_cost and (i % 1000 == 0):
            print('第{0}圈的损失是{1}'.format(i, cost))

    return parameters


def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

     """
    A2 , cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    return predictions


parameters = nn_model(X, Y, 10000, print_cost=False)
# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
predictions = predict(parameters, X)
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')






