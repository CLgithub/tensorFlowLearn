#coding=utf-8

# 有向无环图：inception模块

from keras import models, layers, Input
import numpy as np
import keras

strides=2 	# 每个分支都有相同的步幅2，保证所有分支输出具有相同的尺寸，才能连接在一起，应该使用padding='same'参数
def getModel():
	x = Input(shape=(8, 8, 8),  name='x')
	brance_a = layers.Conv2D(128, 1, activation='relu', strides=strides, padding='same')(x)	# a分支 

	brance_b = layers.Conv2D(128, 1, activation='relu')(x)
	brance_b = layers.Conv2D(128, 3, activation='relu', strides=strides, padding='same')(brance_b)

	brance_c = layers.AveragePooling2D(3, strides=strides, padding='same')(x) 	# 平均池化层
	brance_c = layers.Conv2D(128, 3, activation='relu', padding='same')(brance_c)

	brance_d = layers.Conv2D(128, 1, activation='relu')(x)
	brance_d = layers.Conv2D(128, 3, activation='relu', padding='same')(brance_d)
	brance_d = layers.Conv2D(128, 3, activation='relu', strides=strides, padding='same')(brance_d)

	output = layers.concatenate([brance_a, brance_b, brance_c, brance_d], axis=-1)

	model = models.Model(x, output)
	return model

def getModel_2():
	x = Input(shape=(8, 8, 8),  name='x')
	brance_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)	# a分支 

	brance_b = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
	# brance_b = layers.Conv2D(128, 3, activation='relu')(brance_b)

	brance_c = layers.AveragePooling2D(3, strides=1)(x) 	# 平均池化层
	brance_c = layers.Conv2D(128, 3, activation='relu')(brance_c)

	brance_d = layers.Conv2D(128, 3, activation='relu')(x)
	# brance_d = layers.Conv2D(128, 3, activation='relu')(brance_d)
	brance_d = layers.Conv2D(128, 3, activation='relu', strides=1)(brance_d)

	output = layers.concatenate([brance_a, brance_b, brance_c, brance_d], axis=-1)

	model = models.Model(x, output)
	return model


if __name__ == '__main__':
	model = getModel()
	model.summary()