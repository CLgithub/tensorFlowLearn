#coding=utf-8

import tensorflow as tf
import numpy as np

d=2
xs= (np.fromfunction(lambda a, b:a+b, (10,d)))
xs_test = np.arange(0, 10, 0.5).reshape(10,d)
print('xs-------------\n',xs)
#print('xs-------------\n',xs_test)

#w0 = (np.fromfunction(lambda a, b:a+b, (d,1)))
w0=np.arange(2, 5, 2).reshape(d,1)
print('w0-------------\n',w0)
#b0 = (np.fromfunction(lambda a, b:a+b, (3,d)))
#print('b0-------------\n',b0)
ys=np.matmul(xs,w0)
ys_test=np.matmul(xs_test,w0)
print('ys-------------\n',ys)
print('ys_test-------------\n',ys_test)

x=tf.placeholder('float',[None,d])
w=tf.Variable(tf.zeros([d,1]))
#b=tf.Variable(tf.zeros([d,d]))
#y=tf.nn.softmax(tf.matmul(x,w))
y=tf.matmul(x,w)

y_=tf.placeholder('float',[None,1])

loss=tf.reduce_mean(tf.square(y-y_))
#loss = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train, feed_dict={x:xs, y_:ys})
    #print(step, sess.run(w))
    #print("-"*20)

print(sess.run(w))
print(sess.run(y, feed_dict={x:xs_test}))
    
