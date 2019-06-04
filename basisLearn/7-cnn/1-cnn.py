#condin=utf-8
#第一个卷积神经网络，还需改进才能识别验证码

import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

#定义w变量
def w_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)   #tf.truncated_normal产生随机变量
    return tf.Variable(initial)

#定义b变量
def b_variable(shape):
    initial=tf.constant(0.1,shape=shape)    #常数
    return tf.Variable(initial)

#定义卷积乘函数
def conv2d(x,w):
    #strides=[1,1,1,1],设置卷积步长x和y方向都是1,第一个和最后一个参赛默认是1
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')  #conv2d二维卷积函数,

#定义池化层函数
def max_poo_2x2(x):
    #ksize,设置池化的核函数大小2x2
    #strides，设置池化步长x和y方向都是2
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    #tf.nn.max_pool使用的池化函数

<<<<<<< HEAD
keep_prob=tf.placeholder(tf.float32)

'''********** 搭建cnn神经网络***************'''
def cnn1():
    #定义输入变量
    xs=tf.placeholder(tf.float32,[None,28*28*1])
    ys=tf.placeholder(tf.float32,[None,10])

    x_image=tf.reshape(xs,[-1,28,28,1])

    w_conv1=w_variable([5,5,1,32])  #定义第一层的w，卷积核5x5，rgb=1，输出32个featuremap
    b_conv1=b_variable([32])    #定义第一层的b
    h_conv1=tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)    #构建第一层,padding采用same，输出大小还是30x30,但变厚了30x30x32
    h_pool1=max_poo_2x2(h_conv1)    #池化处理，池化核函数2x2，xy均为2，得到输出15x15x32

    w_conv2=w_variable([5,5,32,64])
    b_conv2=b_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) #15x15x64
    h_pool2=max_poo_2x2(h_conv2)    #8x8x64
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

    w_fc1=w_variable([7*7*64, 1024])
    b_fc1=b_variable([1024])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    h_fc1_dropt=tf.nn.dropout(h_fc1,keep_prob)

    w_fc2=w_variable([1024,10]) 
    b_fc2=b_variable([10])
    aa=tf.matmul(h_fc1_dropt, w_fc2)
    prediction=tf.add(aa, b_fc2)
    return xs,ys,prediction

'''********** 搭建cnn神经网络***************'''
def mycnn():
    #定义输入变量
    xs=tf.placeholder(tf.float32,[None,30*30*1])
    ys=tf.placeholder(tf.float32,[None,9])

    x_image=tf.reshape(xs,[-1,30,30,1])

    w_conv1=w_variable([5,5,1,32])  #定义第一层的w，卷积核5x5，rgb=1，输出32个featuremap
    b_conv1=b_variable([32])    #定义第一层的b
    h_conv1=tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)    #构建第一层,padding采用same，输出大小还是30x30,但变厚了30x30x32
    h_pool1=max_poo_2x2(h_conv1)    #池化处理，池化核函数2x2，xy均为2，得到输出15x15x32

    w_conv2=w_variable([5,5,32,64])
    b_conv2=b_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) #15x15x64
    h_pool2=max_poo_2x2(h_conv2)    #8x8x64
    h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*64])

    w_fc1=w_variable([8*8*64, 1024])
    b_fc1=b_variable([1024])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    h_fc1_dropt=tf.nn.dropout(h_fc1,keep_prob)

    w_fc2=w_variable([1024,9]) 
    b_fc2=b_variable([9])
    prediction=tf.add(tf.matmul(h_fc1_dropt,w_fc2),b_fc2)
    return xs,ys,prediction

xs,ys,prediction=cnn1()
=======

'''********** 搭建cnn神经网络***************'''
'''
#定义输入变量
xs=tf.placeholder(tf.float32,[None,28*28*1])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

x_image=tf.reshape(xs,[-1,28,28,1])

w_conv1=w_variable([5,5,1,32])  #定义第一层的w，卷积核5x5，rgb=1，输出32个featuremap
b_conv1=b_variable([32])    #定义第一层的b
h_conv1=tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)    #构建第一层,padding采用same，输出大小还是30x30,但变厚了30x30x32
h_pool1=max_poo_2x2(h_conv1)    #池化处理，池化核函数2x2，xy均为2，得到输出15x15x32

w_conv2=w_variable([5,5,32,64])
b_conv2=b_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) #15x15x64
h_pool2=max_poo_2x2(h_conv2)    #8x8x64
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

w_fc1=w_variable([7*7*64, 1024])
b_fc1=b_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_dropt=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=w_variable([1024,10]) 
b_fc2=b_variable([10])
aa=tf.matmul(h_fc1_dropt, w_fc2)
prediction=tf.add(aa, b_fc2)

'''
'''********** 搭建cnn神经网络***************'''
#定义输入变量
xs=tf.placeholder(tf.float32,[None,30*30*1])
ys=tf.placeholder(tf.float32,[None,9])
keep_prob=tf.placeholder(tf.float32)

x_image=tf.reshape(xs,[-1,30,30,1])

w_conv1=w_variable([5,5,1,32])  #定义第一层的w，卷积核5x5，rgb=1，输出32个featuremap
b_conv1=b_variable([32])    #定义第一层的b
h_conv1=tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)    #构建第一层,padding采用same，输出大小还是30x30,但变厚了30x30x32
h_pool1=max_poo_2x2(h_conv1)    #池化处理，池化核函数2x2，xy均为2，得到输出15x15x32

w_conv2=w_variable([5,5,32,64])
b_conv2=b_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) #15x15x64
h_pool2=max_poo_2x2(h_conv2)    #8x8x64
h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*64])

w_fc1=w_variable([8*8*64, 1024])
b_fc1=b_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_dropt=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=w_variable([1024,9]) 
b_fc2=b_variable([9])
prediction=tf.add(tf.matmul(h_fc1_dropt,w_fc2),b_fc2)
>>>>>>> 98c50fe14b9fbcf7eaf7bbc360418657d4cab036

cross_entropy=tf.reduce_mean( -tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

def getInput(x):
    xs=np.zeros((x,30*30))
    ys=np.zeros((x,9))
    #img_arr=np.asarray(Image.open('./new_num/0-1'))
    #array1=cv2.imread('./new_num/0-1')
    imgs=os.listdir('./new_num/')
    for i in range(x):
        r=random.randint(0,3248)
        if '.' not in imgs[r]:
            img_arr=np.asarray(Image.open('./new_num/'+imgs[r]))
            img_arr=img_arr.reshape(1,900)
            xs[i]=img_arr

            y=np.zeros(8).tolist()
            y.insert(int(imgs[r][0]),1)
            ys[i]=np.array(y)
    return xs,ys

'''
xss,yss=getInput(1)
print(xss.shape)
print(yss)
batch_xs,batch_ys=mnist.train.next_batch(1)
#print(batch_xs.shape)
#print(batch_ys)

'''
<<<<<<< HEAD

for i in range(1000):
    #batch_xs, batch_ys=getInput(100)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob:0.5})
    if i%50==0:
        #print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        v_xs=mnist.test.images[:1000]
        v_ys=mnist.test.labels[:1000]
        #v_xs, v_ys=getInput(100)
=======
for i in range(1000):
    batch_xs, batch_ys=getInput(100)
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob:0.5})
    if i%50==0:
        #print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        #v_xs=mnist.test.images[:1000]
        #v_ys=mnist.test.labels[:1000]
        v_xs, v_ys=getInput(100)
>>>>>>> 98c50fe14b9fbcf7eaf7bbc360418657d4cab036
        y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
        print(result)
