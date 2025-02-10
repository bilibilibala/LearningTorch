"""
分类问题：数据属于哪一个类别的问题    是根据输入数据将其正确地分配到预定的类别或标签中。
softmax函数：分子是输入信号的指数  分母是所有输入信号指数的和
缺陷：使用指数运算容易溢出
特征：softmax的输出总和=1，可将softmax函数输出的结果称为概率
使用softmax函数 元素的大小位置不会被改变 输出层的softmax可以被省略
"""
import numpy as np

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/ sum_exp_a

    return y


"""
溢出问题  

a=np.array([1010,1000,990])
np.exp(a)/np.sum(np.exp(a))#报错：溢出 RuntimeWarning: overflow encountered in exp
c=np.max(a)
a-c
np.exp(a-c)/np.sum(np.exp(a-c)) #成功输出
"""
# 减去输入信号中的最大值 可解决溢出
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/ sum_exp_a

    return y

a=np.array([0.3,2.9,4.0])
y=softmax(a)
print(y)  #[0.01821127 0.24519181 0.73659691]
print(np.sum(y))  #1.0
