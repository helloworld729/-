import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

# y = tf.constant(36, name='y')
# y_hat = tf.constant(39, name='y_hat')
#
# loss = tf.Variable((y-y_hat)**2, name='loss')
# init = tf.global_variables_initializer()
#
# with tf.Session() as session:  # 注意加括号
#     session.run(init)
#     print(session.run(loss))   # 注意加session.run


# x = tf.placeholder(tf.int64, name='x')
# y = tf.placeholder(tf.int64, name='y')
# with tf.Session() as sess:
#     result = sess.run(tf.multiply(x, y), feed_dict={x: 3, y: 4})
#     print(result)
#     sess.close()


# Y=WX + b ,W和X是随机矩阵，b是随机向量。
# 我们计算WX+b，其中W，X和b是从随机正态分布中抽取的。
# W的维度是（4,3），X是（3,1），b是（4,1）。
# x = tf.constant(np.random.randn(3, 1), name='x')  # 矩阵作为常量
# def linear_function():
#     np.random.seed(1)
#
#     x = np.random.randn(3, 1)
#     w = np.random.randn(4, 3)
#     b = np.random.randn(4, 1)
#
#     y = tf.matmul(w, x) + b
#
#     with tf.Session() as sess:
#         result = sess.run(y)  #
#         sess.close()
#
#     return result


def sigmoid(z):
    x = tf.placeholder(tf.float16, name='x')
    y = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(y, feed_dict={x: z})
        print(result)


def one_hot_matrix(labels,C):
    """
    :param labels: 标签向量--->每一列出现1的位置（从0开始）
    :param C: 深度--->矩阵的行数
    :return: 独热编码
    """
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


def ones(shape):
    """
    print(ones([3, 3]))
    :param shape:
    """
    ones = tf.ones(shape)
    sess = tf.Session()
    result = sess.run(ones)
    sess.close()
    return result


# (1080,64,64,3)  (1,1080)    (120,64,64,3)  (1,120)   [1 2 3 4 5]
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()


# 归一化数据
X_train = X_train_orig / 255
X_test = X_test_orig / 255


# reshape
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T


# 转换为one-hot矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)  # 根据label进行独热编码 行=深度=6=[012345]， 列=训练样本数 =1080
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)    # 根据label进行独热编码 行=深度=6=[012345]， 列=训练样本数 =120


def create_placeholders(n_x,n_y):  # 12288 \ 6
    """
    为TensorFlow会话创建占位符
    参数：
        n_x - 一个实数，图片向量的大小（12288）--->输入节点的数目
        n_y - 一个实数，分类数（从0到5，所以n_y = 6）--->输出节点的数目

    返回：
        X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y - 一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"

    提示：
        None:就是每次训练样本的数目。

    """
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y


# X, Y = create_placeholders(12288, 6)  # X = Tensor("X:0", shape=(12288, ?), dtype=float32)
#
# print("X = " + str(X))
# print("Y = " + str(Y))

# 我们将使用Xavier初始化权重和用零来初始化偏差，比如：
# W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))  # 降到25维
# b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())


def initialize_parameters():  # 12288--25--12--6
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    返回：
        parameters - 包含了W和b的字典

    """

    tf.set_random_seed(1)  # 指定随机种子

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。

# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))  # W1 = <tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))


def forward_propagation(X,parameters):  # [1080 ?]  w1:25*12288
    """
    实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    参数：
        X - 输入数据的占位符，维度为（输入节点数量，样本数量）
        parameters - 包含了W和b的参数的字典

    返回：
        Z3 - 最后一个LINEAR节点的输出

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)        # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)     # Z3 = np.dot(W3,Z2) + b3

    return Z3


# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print("Z3 = " + str(Z3))


def compute_cost(Z3,Y):
    """
    计算成本

    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个占位符，和Z3的维度相同

    返回：
        cost - 成本值


    """
    logits = tf.transpose(Z3) #转置
    labels = tf.transpose(Y)  #转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


# tf.reset_default_graph()

# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost = " + str(cost))


#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
#
# _ , c = sess.run([optimizer,cost],feed_dict={X:mini_batch_X,Y:mini_batch_Y})


def model(X_train,Y_train,X_test,Y_test,
        learning_rate = 0.0001, num_epochs=1000, minibatch_size=32,
        print_cost=True,is_plot=True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    参数：
        X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
        Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
        X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
        Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        mini_batch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图

    返回：
        parameters - 学习后的参数
    """
    ops.reset_default_graph()                # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    n_x = X_train.shape[0];  m = X_train.shape[1]  # 获取输入节点数量12288和样本数1080

    n_y = Y_train.shape[0]                   # 获取输出节点数量6
    costs = []                               # 成本集

    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)      # 输入输入节点12288，输出节点6，返回输出矩阵和输出矩阵的规模[12288 ?] [6 ?]

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)  # X需要压缩为12288 1080

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量
    init = tf.global_variables_initializer()

    # 开始会话并计算
    with tf.Session() as sess:
        # 初始化
        sess.run(init)

        # 正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)    # m = 1080(样本总数)，minibatch的总数量
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:

                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatch  # 一组训练样本的x和y

                # 数据已经准备好了，开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches  # 每一圈epoch平均误差

            #记录并打印成本
            ## 记录成本
            if epoch % 10 == 0:
                costs.append(epoch_cost)
                #是否打印：
                if print_cost and epoch % 100 == 0:
                        print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )
