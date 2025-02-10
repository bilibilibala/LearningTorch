import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#从输入层到第一层的信号传递
#1.
X=np.array([1.0,0.5])  #神经元x1,x2
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])   #对应权重w1,w2
B1=np.array([0.1,0.2,0.3]) #偏置神经元“1”的权重

print(W1.shape)    #(2,3)
print(X.shape)     #(2,)
print(B1.shape)    #(3,)

A1=np.dot(X,W1)+B1   #计算第一层神经元1
#2.
Z1=sigmoid(A1)       #被激活函数转换后的神经元Z

print(A1)    #[0.3 0.7 1.1]
print(Z1)    #[0.57444252 0.66818777 0.75026011]

#从第一层到第二次的信号传递

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

A2=np.dot(Z1,W2)+B2

Z2=sigmoid(A2)

#第2层到输出层的转换
def identity_function(x):
    return x         #输出层的激活函数:恒等函数

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])
A3=np.dot(Z2,W3)+B3

Y=identity_function(A3)

print(Y) #[0.31682708 0.69627909]
