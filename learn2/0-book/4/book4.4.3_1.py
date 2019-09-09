#coding=utf-8

import numpy as np

layer_output=np.array([[2,1],[3,2],[1,3]])
print(layer_output)
#(3,2)
r = np.random.randint(0, high=2, size=layer_output.shape)   #随机生成数值在[0,2)，形状和layer_output一样的张量
print(r)
layer_output *=r    #训练时，按dropout的比率缩小
print(layer_output)

layer_output *= 0.5     # 测试时，要乘以这个比率，假设是0.5
#注意，为了实现这一过程，还可以让两个运算都在训练时进行，而测试时输出保持不变。 这通常也是实践中的实现方式
layer_output /= 0.5     # 成比例放大
