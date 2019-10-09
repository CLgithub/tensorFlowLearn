#coding=utf-8

'''
"Deep learning with Python" 第2章第一节尝试
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


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

#训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
#测试
test_loss,test_acc=network.evaluate(test_images, test_labels)
print(test_loss, test_acc)
#
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predictions=network.predict(test_images)
print(class_names[np.argmax(predictions[1])])
