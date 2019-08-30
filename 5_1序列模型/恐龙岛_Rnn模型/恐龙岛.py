import numpy as np
import time
import matplotlib.pyplot as plt
import cllm_utils


# 获取名称
data = open("dinos.txt", "r", encoding='utf-8').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print(chars)
print("共计有%d个字符，唯一字符有%d个"%(data_size,vocab_size))

char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}

print(char_to_ix)
print(ix_to_char)

def plot_cost(costs):
    """
    输出损失曲线
    :param costs:损失列表
    :return: null
    """
    plt.plot(costs)
    plt.show()

def clip(gradients, maxValue):
    """
    使用maxValue来修剪梯度
    参数：
        gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内
    返回：
        gradients -- 修剪后的梯度
    """
    # 获取参数
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

def sample(parameters, char_to_ix, seed):
    """
    根据训练好的model参数parameters，实际生成字符序列，第一个字符被初始化为全0的‘虚’编码投进去
    停止的条件是产生换行符或者生成的长度4（设定的），实际上最重要的就是参数。
    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典,用于生成序列的最后加\n以及循环停止条件
        seed -- 随机种子        
    返回：
        indices -- 包含推荐字符的索引的 长度为n 的列表。
    """
    
    # 从parameters 中获取参数
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    vocab_size = by.shape[0]  # 也是输入字典的长度 // 2
    n_a = Waa.shape[1]        # 隐层单元的数量
    
    ## 创建独热向量x，初始化为0，作为首个‘成语字’后续会根据输出将其设置为独热编码，即被预测值的索引的位置为1
    x = np.zeros((vocab_size,1))
    
    ## 使用0初始化a_prev记忆值
    a_prev = np.zeros((n_a,1))
    
    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []
    
    # IDX是检测换行符的标志，我们将其初始化为-1。
    idx = -1
    
    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    counter = 0  # 控制随机数种子
    newline_character = char_to_ix["\n"]  # 换行符的索引
    
    while (idx != newline_character and counter < 4):  # 没有检测到换行符并且单词长度小于50
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)  # 下一个时间步的记忆
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)  # 根据输入来预测输出
        
        # 设定随机种子
        np.random.seed(counter + seed)
        
        # 步骤3：根据概率分布，返回最大概率对应的概率索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        
        # 添加到索引中
        indices.append(idx)
        
        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        # 更新a_prev为a
        a_prev = a 
        
        # 累加器
        seed += 1
        counter +=1
    
    if(counter == 4):
        indices.append(char_to_ix["\n"])
    
    return indices


def optimize(X, Y, a0, parameters, learning_rate = 0.01):
    """
    执行训练模型的单步优化，即一个单词序列对应的索引-->单样本整个时间步的训练
    
    参数：
        X -- 整数列表，其中每个整数映射到词汇表中的字符。
        Y -- 整数列表，与X完全相同，但向左移动了一个索引。
        a_prev -- 上一个记忆
        parameters --   权重、偏置字典，包含了以下参数：
                        Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b -- 偏置，维度为(n_a, 1)
                        by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate -- 模型学习的速率    
    返回：
        loss -- 损失函数的值（交叉熵损失）
        gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
        a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """
    
    # 前向传播
    a, y_pred = cllm_utils.rnn_forward(X, a0, parameters)

    # 计算损失
    loss =

    # 反向传播
    gradients = cllm_utils.rnn_backward(X, Y, a, y_pred, parameters)
    
    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients, 5)
    
    # 更新参数
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1], parameters


def model(data, ix_to_char, char_to_ix, num_iterations=3500, 
          n_a=50, dino_names=10, vocab_size=27):
    """
    训练模型并生成恐龙名字
    
    参数：
        data -- 语料库
        ix_to_char -- 索引映射字符字典
        char_to_ix -- 字符映射索引字典
        num_iterations -- 迭代次数
        n_a -- RNN隐层单元数量
        dino_names -- 生成恐龙名字的数量
        vocab_size -- 在文本中的唯一字符的数量
    
    返回：
        parameters -- 学习后了的参数
    """
    costs = []
    # 从vocab_size中获取n_x、n_y
    n_x, n_y = vocab_size, vocab_size
    
    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)
    
    # 初始化损失-->一个实数
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)
    
    # 构建恐龙名称列表
    with open("dinos.txt", encoding='utf-8') as f:
            examples = f.readlines()
            examples = [x.lower().strip() for x in examples]

        # 打乱全部的恐龙名称，返回乱序的恐龙名字列表
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # 初始化记忆为0
    a_prev = np.zeros((n_a,1))
    

    for j in range(num_iterations):        
        index = j % len(examples)  # 达到循环遍历的效果
        X = [None] + [char_to_ix[ch] for ch in examples[index]]  #某单词索引列表
        Y = X[1:] + [char_to_ix["\n"]]  # X左移构成目标索引        
    
        curr_loss, gradients, a_prev, parameters = optimize(X, Y, a_prev, parameters)
        """
        每次投入一个单词对应的X、Yindex序列进行优化
        """       
        
        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        
        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % (100) == 0:
            costs.append(loss)
            # print("第" + str(j+1) + "次迭代，损失值为：" + str(loss))
            # if j % 10000 == 0:
            #     seed = 0
            #     for name in range(dino_names):
            #         # 采样
            #         sampled_indices = sample(parameters, char_to_ix, seed)
            #         cllm_utils.print_sample(sampled_indices, ix_to_char)
            #
            #         # 为了得到相同的效果，随机种子+1
            #         seed += 1
            #     print("\n")
        print("\r当前进度: {:.2f}%".format(j * 100 / num_iterations), end="")
    plot_cost(costs=costs)
    return parameters


#开始时间
start_time = time.clock()
#开始训练
parameters = model(data, ix_to_char, char_to_ix, num_iterations=500, vocab_size=vocab_size)
#结束时间
end_time = time.clock()
#计算时差
minium = end_time - start_time
print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")






