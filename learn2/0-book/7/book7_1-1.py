# coding=utf-8

from keras import models, layers, Input
import numpy as np

seq_model = models.Sequential()
seq_model.add(layers.Dense( 32, activation='relu', input_shape=(64,) ))
seq_model.add(layers.Dense( 32, activation='relu' ))
seq_model.add(layers.Dense( 10, activation='softmax' ))
seq_model.summary()

# 对应的函数式API实现
input_tensor = Input( shape=(64,) )		# 一个张量
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
api_model = models.Model(input_tensor, output_tensor)		# Model类将输入张量和输出张量转换为一个模型
api_model.summary()

api_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

api_model.fit(x_train, y_train, epochs=10, batch_size=128)
score = api_model.evaluate(x_train, y_train)