import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x  # Output layer's activation function: identity function

def init_network():
    #字典（dict）结构来存储网络的权重
    #初始化网络 (init_network)：在这个函数中，我们初始化了网络的权重 (W1, W2, W3) 和偏置 (b1, b2, b3)。
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 输入层到隐藏层1的权重 (2x3)
    network['b1'] = np.array([0.1, 0.2, 0.3])  # 第一层隐藏层的偏置 (3,)

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # 隐藏层1到隐藏层2的权重 (3x2)
    network['b2'] = np.array([0.1, 0.2])  # 第二层隐藏层的偏置 (2,)

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])  # 隐藏层2到输出层的权重 (2x2)
    network['b3'] = np.array([0.1, 0.2])  # 输出层的偏置 (2,)

    return network

def forward(network, x):
    #封装将输入信号转换为输出信号的处理过程
    #字典的结构让你可以使用 network['W1'] 这样的语法来直接访问网络中的权重矩阵和偏置项
    #前向传播 (forward)：从输入到输出方向的传播。这个函数计算输入 x 通过网络的每一层传播过程。它首先计算每一层的加权输入（np.dot），然后通过激活函数（sigmoid）进行激活，最后经过输出层使用恒等函数（identity_function）。
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1      # 计算第一层的加权输入
    z1 = sigmoid(a1)             # 第一层激活函数（sigmoid）
    a2 = np.dot(z1, W2) + b2     # 计算第二层的加权输入
    z2 = sigmoid(a2)             # 第二层激活函数（sigmoid）
    a3 = np.dot(z2, W3) + b3     # 计算第三层的加权输入
    y = identity_function(a3)    # 输出层的恒等函数

    return y
#forward在此结束

network = init_network()  # 初始化网络
x = np.array([0.1, 0.5])  # 输入数据
y = forward(network, x)   # 进行前向传播
print(y)   #[0.31234736 0.6863161 ]
