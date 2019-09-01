import numpy as np

def one_hot(index_list, deep):
    return np.eye(deep)[:,index_list]



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    e_x = np.exp(-x)
    return 1 / (1 + e_x)


def smooth(loss, cur_loss):
    # 指数平均加权
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    # 索引-->字符-->单词
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    ba = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    return parameters


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def rnn_cell_forward(xt, a_prev, parameters):
    """
    单步传播
    Arguments:
    xt --     t时间步样本切片   (n_x, m).
    a_prev -- m个样本的记忆列表 (n_a, m)
    parameters -- 权重、偏置参数
                        Wax -- x-->a (n_a, n_x)
                        Waa --       (n_a, n_a)
                        Wya -- a-->y (n_y, n_a)
                        ba --        (n_a, 1)
                        by --        (n_y, 1)
    Returns:
    a_next -- 输出记忆
    yt_pred -- 单步softmax预测值
    cache -- 输出记忆、预测值
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    return a_next, yt_pred.squeeze()

def rnn_forward(x, y, a0, parameters):
    """
    前向
    Arguments:
    x -- 多样本整个时间步 (n_x, m, T_x).
    a0 -- 初始化的所有样本的记忆列表 (n_a, m)
    parameters -- 权重、偏置参数
                        Wax -- x-->a (n_a, n_x)
                        Waa --       (n_a, n_a)
                        Wya -- a-->y (n_y, n_a)
                        ba --        (n_a, 1)
                        by --        (n_y, 1)
    Returns:
    a -- 所有样本整个时间步的记忆张量 (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    n_x, T_x = x.shape  # 输入unit数量、样本数、时间步总数
    n_y, n_a = parameters["Wya"].shape  # 输出节点、隐层节点的数目
    a = np.zeros((n_a, T_x))  # 初始化记忆与预测张量
    y_pred = np.zeros((n_y, T_x))
    a_next = a0.reshape(-1, 1)


    for t in range(T_x):
        x_t = x[:, t].reshape(-1, 1)  # 若x.shape = （a,b,c）则x_t的shape = （a,b）--->(a,b,1)
        y_t = y[:, t].reshape(-1, 1)
        # a:编码长度 b:样本数目 c:步长-->样本中序列长度
        a_next, yt_pred = rnn_cell_forward(x_t, a_next, parameters)  # 单步前向
        a[:, t] = a_next.squeeze()  # 更新t时间步的记忆值
        y_pred[:, t] = yt_pred

    loss = -np.sum(y * np.log(y_pred))

    return loss, a, y_pred


def rnn_cell_backward(dz, gradients, parameters, x_t, a_next, a_prev):
    """
    符号约定：
            Z = Wya * a_nest + by
            y_hat = softmax(Z)
            so : dZ = y_hat - y
            and: da_next = np.dot(dZ, Wya.T)
    :param dz: 基本梯度
    :param gradients: 梯度字典，因为权值共享，所以一个序列累计的更改梯度
    :param parameters: 参数字典
    :param x: 当前时间步的输入，即X<t>
    :param a_next: 提供给下一序列的记忆
    :param a_prev: 前一序列的记忆
    :return: 梯度字典，在同一序列进行中，自动的累加参数的梯度，在不同序列之间，先经过梯度修剪-->根据学习率更新参数
    """
    gradients['dWya'] += np.dot(dz, a_next.T)
    gradients['dby'] += np.sum(dz)
    da = np.dot(parameters['Wya'].T, dz)
    daraw = (1 - a_next * a_next) * da
    gradients['db'] += np.sum(daraw)
    gradients['dWax'] += np.dot(daraw, x_t.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def rnn_backward(x, y, a, y_pred, parameters):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.
    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    n_a, T_x = a.shape
    gradients = {}

    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(ba), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like([n_a, 1])

    total_step = T_x

    for t in reversed(range(1, total_step, 1)):
        x_t = x[:, t]
        dz = y_pred[:, t] - y[:, t]
        a_next = a[:, t]
        a_prev = a[:, t - 1]
        gradients = rnn_cell_backward(dz.reshape(-1, 1), gradients, parameters, x_t.reshape(-1, 1), a_next.reshape(-1, 1), a_prev.reshape(-1, 1))

    return gradients






