#coding=utf-8
'''
基础学习：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
可以和1-.py结合相看，两种写法
'''
import tensorflow as tf
import numpy as np

#构建数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#搭建模型
x=tf.placeholder(tf.float32, [None])
y=tf.placeholder(tf.float32, [None])
weights=tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y_model=weights*x+biases


#计算误差
loss = tf.reduce_mean(tf.square(y-y_model))

#用梯度下降法传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train,feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        w_run=sess.run(weights)
        b_run=sess.run(biases)
        loss_run=sess.run(loss, feed_dict={x:x_data, y:y_data})
        print(step,w_run,b_run,loss_run)
