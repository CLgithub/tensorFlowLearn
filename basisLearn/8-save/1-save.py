#coding=utf-8
'''
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-06-save/
保存与读取训练结果
'''
import tensorflow as tf
import numpy as np

keep_prob=tf.placeholder(tf.float32)

#构建数据集
d=1
t=d*100
x_data = np.linspace(-1, 1, t, dtype=np.float32).reshape(100,d) #在-1～1之间取100个点，然后变成100行1列的矩阵
noise = np.random.normal(0, 0.15, x_data.shape).astype(np.float32)
#y_data = np.square(x_data) - 0.5 + noise    #square 计算各元素的平方
y_data =5*x_data - 0.5 + noise    #square 计算各元素的平方
# y=x^2-0.5+noise

def myNet():
    xs=tf.placeholder(tf.float32, [None, d])
    ys=tf.placeholder(tf.float32, [None, d])
    #weights=tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    weights = tf.Variable(tf.random_normal([d, d])) #指定W的名称
    biases = tf.Variable(tf.zeros([1, d])+0.1)
    wx_plus_b = tf.matmul(xs, weights)+biases 
    #y_model=weights*xs+biases
    return wx_plus_b,xs,ys,weights,biases

prediction,xs,ys,w,b=myNet()
#prediction,xs,ys=myNet2()

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
saver=tf.train.Saver()

'''*************** 计算 ****************'''
def learn1():
    for i in range(200):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 20 == 0:
            loss_run=sess.run(loss, feed_dict={xs: x_data, ys:y_data })
            w_run=sess.run(w)
            b_run=sess.run(b)
            print(loss_run)
            print(w_run)
            print(b_run)
            print('------------------')
            if loss_run<0.025:
                saver.save(sess,"my_net/save_net.ckpt") #存储



#learn1()
#读取
saver.restore(sess,"my_net/save_net.ckpt")
print(sess.run(w))
