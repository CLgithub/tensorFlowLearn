#coding=utf-8

# 共享层权重

from keras import models, layers, Input
import numpy as np
import keras


def getModel():
	lstm = layers.LSTM(32)	# 实例化一个LSTM层，32个处理神经元

	left_input = Input(shape=(None, 128))
	left_output = lstm(left_input)	# 第一次使用lstm

	right_input = Input(shape=(None, 128))
	right_output = lstm(right_input)	# 第二次使用lstm

	merged = layers.concatenate([left_output, right_output], axis=-1)
	predictions = layers.Dense(1, activation='sigmoid')(merged)

	model = models.Model([left_input, right_input], predictions)
	return model


if __name__ == '__main__':
	model = getModel()
	model.summary()