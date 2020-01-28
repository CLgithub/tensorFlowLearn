#coding=utf-8

# 有向无环图：残差连接

from keras import models, layers, Input
import numpy as np
import keras


# 恒等残差连接
def getModel():
	x = Input(shape=(8, 8, 128),  name='x')
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)

	y = layers.add([y, x])	# 将原始x与输出特征相加
	
	model = models.Model(x, y)
	return model

# 线性残差连接
def getModel_2():
	x = Input(shape=(8, 8, 128),  name='x')
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)	
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2), strides=2)(y)
	# print(y.shape)	# y的形状，(?, 4, 4, 128)
	# print(x.shape)	# x的形状，(?, 8, 8, 128)

	residual = layers.Conv2D(128, (1,1), strides=2, padding='same')(x)	# 使用 1×1 卷积，将原始x线性下采样与y的形状相同

	y = layers.add([y, residual])	# 将原始x与输出特征相加
	
	model = models.Model(x, y)
	return model


if __name__ == '__main__':
	model = getModel_2()
	model.summary()