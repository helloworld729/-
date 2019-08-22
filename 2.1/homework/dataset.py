import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
def load_dataset(is_plot=True):
    """
    返回圆形散点图，每一行为一个样本，所以坐标网格由0列和1列生成，
    label为0或者1
    :param is_plot: 是否打印图形
    :return: 训练、验证数据
    """
    np.random.seed(1)
    train_x, train_y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
    np.random.seed(2)
    test_x, test_y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)

    if is_plot:
        plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    train_x = train_x.T
    test_x = test_x.T
    train_y = train_y.reshape(1, -1)
    test_y = test_y.reshape(1, -1)
    return train_x, train_y, test_x, test_y


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def initial_parameters(layer_dims):
    """
    初始化权重和偏置
    :param layer_dims:各层的神经元数目
    :return: 初始化后的参数
    """
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def forward_propgation_dorpout(x, parameters, keep_prob):
    """
    带随机‘删除’节点的前向传播
    :param x:         数据集
    :param parameters:参数集
    :param keep_prob: 保留概率
    :return:          估计值、缓存
    """
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    k1 = np.random.rand(a1.shape[0], a1.shape[1]) <= keep_prob
    a1 = a1 * k1 / keep_prob
    # a1 = a1 * k1

    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    k2 = np.random.rand(a2.shape[0], a2.shape[1]) <= keep_prob
    a2 = a2 * k2 / keep_prob
    # a2 = a2 * k2

    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, w1, b1, k1, z2, a2, w2, b2, k2,  z3, a3, w3, b3)

    return a3, cache


def cal_cost(a3, y):
    m = a3.shape[1]
    cross_entropy = np.multiply(np.log(a3), y) + np.multiply(np.log(1-a3), 1-y)
    cost = -1/m * np.nansum(cross_entropy)
    return cost


def back_with_lambd(x, y, cache, lambd):
    (z1, a1, w1, b1, k1, z2, a2, w2, b2, k2,  z3, a3, w3, b3) = cache

    m = x.shape[1]  # 每一列表示一个样本

    dz3 = 1/m * (a3 - y)
    dw3 = np.dot(dz3, a2.T) + (lambd/m * w3)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3)
    # dz2 = np.multiply(da2, np.int64(a2>0))
    dz2 = np.multiply(da2, a2 > 0)
    dw2 = np.dot(dz2, a1.T) + (lambd/m * w2)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    dz1 = np.multiply(da1, a1 > 0)
    dw1 = np.dot(dz1, x.T) + (lambd/m * w1)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    grads = {}
    grads['dw1'] = dw1; grads['db1'] = db1
    grads['dw2'] = dw2; grads['db2'] = db2
    grads['dw3'] = dw3; grads['db3'] = db3
    return grads


def back_with_keeprob(x, y, cache, keep_prob):
    (z1, a1, w1, b1, k1, z2, a2, w2, b2, k2,  z3, a3, w3, b3) = cache

    m = x.shape[1]  # 每一列表示一个样本

    dz3 = 1/m * (a3 - y)
    dw3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3) * k2 / keep_prob
    # da2 = np.dot(w3.T, dz3) * k2
    dz2 = np.multiply(da2, a2 > 0)

    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2) * k1 / keep_prob
    # da1 = np.dot(w2.T, dz2) * k1
    dz1 = np.multiply(da1, a1 > 0)
    dw1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    grads = {}
    grads['dw1'] = dw1; grads['db1'] = db1
    grads['dw2'] = dw2; grads['db2'] = db2
    grads['dw3'] = dw3; grads['db3'] = db3
    return grads


def update_para(grads, parameters, learnint_rate):
    for l in range(1, parameters.__len__() // 2):
        parameters['w' + str(l)] = parameters['w' + str(l)] - learnint_rate * grads['dw' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learnint_rate * grads['db' + str(l)]
    return parameters


def plt_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (*1000)')
    plt.title('learnint_rate = ' + str(learning_rate))
    plt.show()


def model(x, y, learning_rate=0.3, iterations=30000, print_cost=False, is_plot=False, plot_boundary=False, lamdb=0, keep_prob=1):
    """
    定义模型：初始化（3种方式）-前向（是否dropout？）-后向（是否正则化、是否dropout）
    3层网络：2-3-1  rulu，relu，sigmoid
    :param x:              训练数据
    :param y:              训练label
    :param learning_rate:  学习率
    :param iterations:     圈数
    :param print_cost:     是否打印cost值
    :param is_plot:        是否打印cost曲线
    :param lamdb:          正则化超参数
    :param keep_prob:      节点保留概率
    :return:               训练后的参数
    """
    costs = []
    layer_dims = [x.shape[0], 20, 3, 1]
    parameters = initial_parameters(layer_dims)  # 返回3组权重和偏置
    for iteration in range(iterations):
        # 前向传播
        a3, cache = forward_propgation_dorpout(x, parameters, keep_prob=keep_prob)

        # 计算成本
        cost = cal_cost(a3, y)
        if iteration % 2000 == 0:
            costs.append(cost)
            if print_cost:
                print('第{0}圈， 损失是：{1}'.format(iteration, cost))

        # 反向传播
        if keep_prob == 1:
            grads = back_with_lambd(x, y, cache, lambd=lamdb)
        else:
            grads = back_with_keeprob(x, y, cache, keep_prob=keep_prob)

        # 更新参数
        parameters = update_para(grads, parameters, learning_rate)

        if iteration%10 == 0 and plot_boundary:
            plt.title("Model with L2-regularization")
            axes = plt.gca()
            # axes.set_xlim([-0.75,0.40])
            # axes.set_ylim([-0.75,0.65])
            plot_decision_boundary(parameters, x, y)

    if is_plot:
        plt_cost(costs, learning_rate)
    return parameters


def predict(x, y, parameters):
    a3, cache = forward_propgation_dorpout(x, parameters, keep_prob=1)
    a3 = a3 > 0.5
    p = np.ones((a3.shape[0], a3.shape[1]))
    p = p * a3
    return np.mean(p[0, :] == y[0, :])


def plot_decision_boundary(parameters, x, y):
    # 设定坐标范围
    x_min, x_max = x[0, :].min() - 0.2, x[0, :].max() + 0.2  # 图形裕量
    y_min, y_max = x[1, :].min() - 0.2, x[1, :].max() + 0.2
    h = 0.01  # 数据坐标的step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    """
    生成同维度的矩阵xx， yy 其中行数由y间距和步进值确定，列数由x间距和步进值确定。
    相同位置的数据构成某一坐标点
    """

    Z = predict_dec(parameters, np.c_[xx.ravel(), yy.ravel()].T)  # z就是调用时候的 lambda  ,c_后把两个矩阵左右并置
    """
    首先将xx与yy降成一维向量，如果没有rehsape的话，默认生成秩为1的向量-->列向量
    np.c将两个列向量并置-->shape=（m，2）-->转置为（2，m）  # m就是坐标点的数目
    用训练生成的参数对每个点预测，生成（1，m）预测结果
    将预测结果reshape成坐标网格的维度
    调用contourf函数显示
    """
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[0, :], x[1, :], c=y.squeeze(), cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, x):
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propgation_dorpout(x, parameters, 1.0)
    predictions = (a3 > 0.5)
    return predictions