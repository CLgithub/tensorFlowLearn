#coding=utf-8
'''
利用tensorflow 纯手工编写神经网络完成基本分类
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

#导入数据集
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#print(train_images.shape)
#print(train_images[:100].shape)

#定义添加层函数
def addlayer(inputs, in_d, out_d, activation_function=None):
    w=tf.Variable(tf.random_normal([in_d, out_d]))
    b=tf.Variable( tf.zeros([1, out_d])+0.1 )
    wx_b=tf.matmul(inputs, w)+b
    if activation_function is None:
        outputs=wx_b
    else:
        outputs=activation_function(wx_b)
    return outputs

#构建神经网络
xs=tf.placeholder("float",[None, 784])
ys=tf.placeholder("float",[None, 10])
def setLayer1():
    l1=addlayer(xs, 784, 10, activation_function=tf.nn.softmax)
    return l1
def setLayer2():
    l1=addlayer(xs, 784, 10, activation_function=None)
    #l2=addlayer(l1, 128, 10, activation_function=None)
    return l1
ys_=setLayer1()

#计算误差
loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(ys_), reduction_indices=[1]))
#反向传播 随机梯度下降
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#开始训练
for i in range(1000):
    ii=random.randint(0, 5)
    #batch_xs, batch_ys=fashion_mnist.train.next_batch(100)
    bx=train_images[ii*100:(ii+1)*100]
    by=train_labels[ii*100:(ii+1)*100]
    bx=bx.reshape(100,28*28)
    bx=bx/255
    bys=[]
    for b in by:
        by_=[0,0,0,0,0,0,0,0,0]
        by_.insert(b,1)
        bys.append(by_)
    bys=np.array(bys)
    sess.run(train_step, feed_dict={xs:bx, ys:bys})
    if i % 50 == 0:
        loss_run=sess.run(loss, feed_dict={xs:bx, ys:bys})
        cp=tf.equal(tf.argmax(ys,1),tf.argmax(ys_,1))   #比较两个数组是否相等,相等则改位置上为1
        acc=tf.reduce_mean(tf.cast(cp,"float")) #然后计算平均值
        acc_run=sess.run(acc, feed_dict={xs:bx,ys:bys})
        print('损失：',loss_run,'准确率：',acc_run)
