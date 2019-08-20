#coding=utf-8

import tensorflow as tf
#from tensorflow import keras
from keras import layers
from keras import models
import numpy as np

'''
3.1.1 层:深度学习的基本组建
'''

arr1=np.array([[1,3],[3,3],[1,2]])
print(arr1)
print(arr1.shape)

arr2=np.array([1,3,3,3,1,2])
print(arr2)
print(arr2.shape)
'''
layers.Dense(c, input_shape=(a,))
设置一个全连接层，输入张量的形状为(samples,a)，其中samples是默认会加上的，是批量维度，可以是任意值，接收任意多个数据，返回相同数量的张量，张量形状变成了(samples,c)，不能写成 layers.Dense(c, input_shape=(a))，因为必须要让a表示为形状，而不是int
'''
model=models.Sequential()
model.add( layers.Dense( 32, input_shape=(784,) ) )   #如何证明这里只能接收 第一个维度大小为784的2D张量作为输入,
model.add( layers.Dense( 12, input_shape=(33,2) ) )

print(model)
