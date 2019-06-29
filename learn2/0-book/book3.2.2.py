#coding=utf-8

from keras import models
from keras import layers

'''
3.2.2 使用Keras开发：概述
    定义模型有两种方法：
        1.利用Sequential类
        2.函数式API

'''

# 1.利用Sequential类
model1=models.Sequential()
model1.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model1.add(layers.Dense(10, activation='softmax'))
print(model1)

# 2.函数式API
input_tensor=layers.Input(shape=(784,))
x=layers.Dense(32, activation='relu')(input_tensor)
output_tensor=layers.Dense(10, activation='softmax')(x)
model2=models.Model(inputs=input_tensor, outputs=output_tensor)
print(model2)
#利用函数式API，你可以操作模型处理的数据张量，并将层应用于这个张量，就好像这些层式函数一样
