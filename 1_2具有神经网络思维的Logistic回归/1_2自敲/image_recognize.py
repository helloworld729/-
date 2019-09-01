import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

"""
0、约定：所有的向量都是列向量
1、加载图片文件
2、获取图片文件参数（训练集、测试集的图片数目、图片的尺寸）
3、图片文件向量209*64*64*3转化为（64*64*3）*209（图片设置为列向量）
4、数据集规范化
5、定义sigmoid函数
6、定义权重初始化函数
7、前向传播函数，返回步进值字典和损失lost
8、反向优化函数，返回最终的权重和b
9、test集预测函数
10、7、8、9的综合
"""


def prepare():
    x_train, y_train, x_test, y_test, label = load_dataset()  # (209, 64, 64, 3),(1, 209),(50, 64, 64, 3),(1, 50),[b'non-cat' b'cat']
    train_picture_num = x_train.shape[0]  # 209
    test_picture_num = x_test.shape[0]  # 50
    x_train_compress = x_train.reshape(train_picture_num, -1).T  # (12288, 209)
    x_test_compress = x_test.reshape(test_picture_num, -1).T  # (12288, 50)
    x_train_compress = x_train_compress / 255
    x_test_compress = x_test_compress / 255
    prep = dict()
    prep['X_train'] = x_train_compress; prep['Y_train'] = y_train
    prep['X_test'] = x_test_compress; prep['Y_test'] = y_test
    prep['num_train'] = train_picture_num
    prep['num_test'] = test_picture_num
    return prep


def sigmoid(z):
    """can be a real num, list, np.array"""
    return 1 / (1 + np.exp(-z))


def weight_initialize(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b


def propagation(w, b, X, Y,):
    Z = np.dot(w.T, X) + b  # row vector
    A = sigmoid(Z)
    dZ = A - Y
    dW = (1/X.shape[1]) * np.dot(X, dZ.T)
    db = (1/X.shape[1]) * np.sum(dZ)
    lost = (-1/X.shape[1]) * np.sum(Y*np.log(A)+(1-Y)*np.log(1 - A))
    # print(lost)
    step = dict()
    step['dw'] = dW
    step['db'] = db
    return step, lost


def test(w_final, b_final, X_test, Y_test, train = False):
    Z = np.dot(w_final.T, X_test) + b_final  # row vector
    A = sigmoid(Z)
    Y_prediction = np.zeros((1, X_test.shape[1]))
    for i in range(X_test.shape[1]):
        Y_prediction[0, i] = 0 if A[0, i] < 0.5 else 1
    # print('Y_test:{}'.format(Y_test))
    # print('Y_prediction:{}'.format(Y_prediction))
    percentage = 1 - np.mean(np.abs(Y_test - Y_prediction))
    # if train:
    #     print('训练集正确率：{}%'.format(percentage*100))
    # else:
    #     print('测试集正确率：{}%'.format(percentage*100))
    return percentage


def optimize(iteration=1500, learnint_rate=0.01, print_cost=False):
    """迭代n圈,每一圈中：正向调用拿到dw，db-->更新w，b"""
    prep = prepare()
    train_percentage = list(); test_percentage = list()
    cost = list()
    X, Y = prep['X_train'], prep['Y_train']
    w, b = weight_initialize(X.shape[0])
    X_test, Y_test = prep['X_test'], prep['Y_test']

    for iter in range(iteration):
        step, lost = propagation(w, b, X, Y,)
        w = w-learnint_rate*step['dw']
        b = b-learnint_rate*step['db']

        if (iter % 100 == 0):
            cost.append(lost)
            test_percentage.append(test(w, b, X_test, Y_test, train=False))
            train_percentage.append(test(w, b, X, Y, train=True))
            if print_cost:
                print('迭代第{0}圈，损失{1}'.format(iter, cost))

    paremeter = dict()
    paremeter['train_percentage'] = train_percentage
    paremeter['test_percentage'] = test_percentage
    paremeter['cost'] = cost
    return paremeter

def percentage_curve(train_percentage, test_percentage):
    plt.plot(train_percentage, label='train')
    plt.plot(test_percentage, label='test')
    legend = plt.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # plt.legend()
    plt.xlabel('iteration*100')
    plt.ylabel('right_percentage')
    plt.title('right_percentage with different iterations')
    plt.show()


def cost_curve():
    learn_rate = [0.01, 0.001, 0.005]
    label = ['rate = 0.01', 'rate = 0.001', 'rate = 0.005']
    for i in range(len(learn_rate)):
        para = optimize(learnint_rate=learn_rate[i])
        cost = para['cost']
        plt.plot(cost, label=label[i])
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.xlabel('iterations * 100')
    plt.ylabel('cost')
    plt.title('cost with different iterations')
    plt.show()


cost_curve()
