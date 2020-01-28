#coding=utf-8

# 将模型作为层

from keras import models, layers, Input, applications
import numpy as np
import keras


def getModel():
	xception_base = applications.Xception(weights=None, include_top=False)	# 使用图像处理基础模型Xception，不包括顶部

	left_input = Input(shape=(250, 250, 3))
	right_input = Input(shape=(250, 250, 3))

	left_features = xception_base(left_input)	# 将模型xception_base当作层来使用
	right_features = xception_base(right_input)

	merged_features = layers.concatenate([left_features, right_features], axis=-1)

	model=models.Model([left_input,right_input], merged_features)

	return model

def getModel_test():
	x = Input(shape=(8, 8, 128),  name='x')
	y = layers.Dense(1)(x)
	model1 = models.Model(x, y)	# 得到模型model1

	z = Input(shape=(8, 8, 128),  name='x')
	output = model1(z)		# 将模型model1当真层来使用

	models2=models.Model([x,z], output)
	models2.summary()


if __name__ == '__main__':
	model = getModel()
