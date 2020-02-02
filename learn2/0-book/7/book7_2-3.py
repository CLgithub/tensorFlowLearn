#coding=utf-8

# 自定义回调函数：在每轮结束后将模型每层的激活保存到硬盘

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import os, shutil
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input


#导入数据
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#设置神经网络
network=models.Sequential()
network.add(layers.Dense( 512, activation='relu', input_shape=(28*28,) )) #添加一个全连接层，512个神经元
network.add(layers.Dense( 10 ,activation='softmax' ))

# print(network.summary())

#编译神经网络
network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
)

#数据预处理
train_images=train_images.reshape((60000, 28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000, 28*28))
test_images=test_images.astype('float32')/255
#对标签进行分类编码，第3张会详细解释
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


# 自定义回调函数
class ActivationLogger(keras.callbacks.Callback):

	def set_model(self, model):
		self.model=model 	# 在训练之前由父模型调用，告诉回调函数是哪个模型在调用它
		layer_outputs = [layer.output for layer in model.layers]	# 得到每一层输出列表
		self.activations_model = keras.models.Model(model.input, layer_outputs)	#模型输入与输出列表，组成新的模型

	def on_epoch_end(self, epoch, logs=None):	# 每轮训练结束时，这些方法倍调用时都有个logs参数，
		if self.validation_data is None:
			raise runtimeerror('requires validation_data.')
		validation_sample = self.validation_data[0][0:1]	# 得到验证数据的第一个输入样本
		activations = self.activations_model.predict(validation_sample)	# 将样本数据输入到模型，得到每一层输出列表
		f = open('activations_at_epoch_'+str(epoch)+'.npz', 'wb')
		np.savez(f, activations[0])
		f.close()

def getCallbackslist():
	# 
	callbacks_list = [
		ActivationLogger()
	]

	return callbacks_list

callbacks_list = getCallbackslist()
#训练
network.fit(
		train_images, train_labels, 
		epochs=5, 
		batch_size=128,
		validation_data=(test_images, test_labels),
		callbacks=callbacks_list
	)
#测试
test_loss,test_acc=network.evaluate(test_images, test_labels)
print(test_loss, test_acc)
#
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predictions=network.predict(test_images)
print(class_names[np.argmax(predictions[1])])

