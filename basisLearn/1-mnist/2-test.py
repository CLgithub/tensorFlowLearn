#coding=utf-8

import tensorflow as tf
import numpy as np

#构建测试集
xs = np.random.rand(1000).astype(np.float32) #随机获取数据
xs_test = np.random.rand(10).astype(np.float32) #随机获取数据
print('xs-------------\n',xs)
ys = xs*3+2
ys_test = xs_test*3+2
print('ys-------------\n',ys)
print('ys_test-------------\n',ys_test)

#构建数据模型
x=tf.placeholder('float',[None])
w=tf.Variable(tf.zeros([1]))
b=tf.Variable(tf.zeros([1]))
y=w*x + b

y_=tf.placeholder('float',[None])

#误差计算，并用随机梯度下降反向传播
loss=tf.reduce_mean(tf.square(y-y_))
train= tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#训练
for step in range(100):
    sess.run(train, feed_dict={x:xs, y_:ys})
    #print(step, sess.run(w), sess.run(b))

print(sess.run(y, feed_dict={x:xs_test})) #把xs_test放进去，计算得到y，看与正确的ys_test相差多大
correct_prediction = tf.equal(y,y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print(accuracy)
print(sess.run(accuracy, feed_dict={x:xs_test, y_:ys_test}))


