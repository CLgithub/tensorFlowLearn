#coding=utf-8
# 简单卷积神经网络与密集连接网络效果对比,book2.1.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


# 导入数据
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 设置卷积神经网络
model = models.Sequential()
#添加一个卷积层，接收图片形状(28,28,1),通道数量为32，小窗大小为(3,3)，激活函数为relu    ,通道数--深度轴
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))	
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu' ))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu' ))
#下一步是将最后的输出张量(3,3,64)传给一个密集连接分类器中，但输出是1D的，所以需要拉平，可以尝试在不同的地方拉平
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())	#打印网络结构

#编译神经网络
model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
)

#数据预处理
train_images=train_images.reshape((60000, 28, 28, 1))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000, 28, 28, 1))
test_images=test_images.astype('float32')/255
#对标签进行分类编码，第3张会详细解释
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#训练
model.fit(train_images, train_labels, epochs=5, batch_size=64)
#测试
test_loss,test_acc=model.evaluate(test_images, test_labels)
print(test_loss, test_acc)
#
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predictions=model.predict(test_images)
print(class_names[np.argmax(predictions[1])])
