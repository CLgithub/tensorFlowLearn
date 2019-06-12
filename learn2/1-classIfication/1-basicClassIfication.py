#coding=utf-8

'''
该学习内容源自tensorflow官网教程，在对人工智能、机器学习、深度学习、神经网络、tensorflow、keras的关系有了进一步的了解后，再次学习该教程

https://www.tensorflow.org/tutorials/keras/basic_classification
'''

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Hei']  #指定显示的中文字体
plt.rcParams['axes.unicode_minus']=False  #用来正常显示中文标签
#有中文出现的情况，需要u'内容'

#导入数据集
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = [u'T恤衫', u'裤子', u'套头毛衣', u'连衣裙', u'外套', u'凉鞋', u'衬衫', u'运动鞋', u'包包', u'踝靴']
#预处理，不知道为何这么做
train_images=train_images/255.0
test_images=test_images/255.0

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.figure(figsize=(10,10)) #开一个小窗口
for i in range(25):
    plt.subplot(5,5,i+1)    #甚至横竖行
    #plt.xticks([])          #设置不显示像素横
    #plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i]) #cmap=plt.cm.binary 灰度图显示
    plt.xlabel(class_names[train_labels[i]]) #设置在横方向上显示标题
plt.show()
'''

def createModel():
    #设置层，构建模型
    model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),          #L1,先将数据拉平,28*28=784
        keras.layers.Dense(128, activation=tf.nn.relu),      #L2,将拉平后的数据传给L2层128个神经元，使用relu进行激励
        keras.layers.Dense(10, activation=tf.nn.softmax)    #L3结果层,将L2层的结果传给10个神经元，激励函数为softmax，得到具有10个概率得分的数组
    ])
    #编译模型
    model.compile(
        optimizer=tf.train.AdamOptimizer(), #设置优化器
        loss='sparse_categorical_crossentropy', #设置损伤函数
        metrics=['accuracy']    #设置监控指标=精准度
    )
    return model

#训练模型
def learn1(model):
    model.fit(train_images, train_labels, epochs=5)
    return model


model=createModel()
model=learn1(model)
'''
test_loss,test_acc=model.evaluate(test_images, test_labels)
print(test_acc)
'''

predictions=model.predict(test_images)
print(class_names[np.argmax(predictions[1])])
