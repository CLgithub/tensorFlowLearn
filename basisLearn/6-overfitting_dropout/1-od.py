#coding=utf-8
'''
基础学习：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
可以和1-.py结合相看，利用dropout解决overfitting问题
'''
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
#from sklearn.cross_validation import train_test_split  #已抛弃
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#构建数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*10.1+420.3

#搭建模型
with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32, [None], name='x_in')
    y=tf.placeholder(tf.float32, [None], name='y_in')


with tf.name_scope('layer1'):
    keep_prob=tf.placeholder(tf.float32)    #保存概率

    ran=tf.random_uniform([1], -1.0, 1.0, name='ran')
    tf.summary.histogram('layer1-ran',ran)

    weights=tf.Variable(ran, name='w')
    tf.summary.histogram('layer1-w',weights)

    biases = tf.Variable(tf.zeros([1]), name='b')
    tf.summary.histogram('layer1-b',biases)

    with tf.name_scope('wx_plus_b'):
        y_model=weights*x+biases
        y_model=tf.nn.dropout(y_model, keep_prob)   #添加随机失活层
        tf.summary.histogram('layer1-wx_plus_b',y_model)


#计算误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-y_model))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #用梯度下降法传播误差
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#训练
sess = tf.Session()
merged=tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

for step in range(200):
    sess.run(train,feed_dict={x:x_data, y:y_data, keep_prob:0.8})
    if step % 20 == 0:
        merged_run=sess.run(merged, feed_dict={x:x_data, y:y_data, keep_prob:0.8})
        writer.add_summary(merged_run, step)

        w_run=sess.run(weights)
        b_run=sess.run(biases)
        loss_run=sess.run(loss, feed_dict={x:x_data, y:y_data, keep_prob:0.8})
        print(step,w_run,b_run,loss_run)
