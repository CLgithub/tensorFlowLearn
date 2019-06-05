#coding=utf-8
'''
神经网络可以用tensorflow一层一层搭建，但也可以用keras快速简洁搭建，之前做的衣物分类和足球赔率均用的keras，学完这教程后应切换到官方教程学习,使用不同的网络结构来处理一个问题
'''

import tensorflow as tf
import numpy as np
import keras

#构建数据集
d=3
t=d*100
x_data = np.linspace(-1, 1, t, dtype=np.float32).reshape(100,d,1) #在-1～1之间取100个点，然后变成100行1列的矩阵
print(x_data)
noise = np.random.normal(0, 0.15, x_data.shape).astype(np.float32)
#y_data = np.square(x_data) - 0.5 + noise    #square 计算各元素的平方
y_data =5*x_data - 0.5 + noise    #square 计算各元素的平方
# y=x^2-0.5+noise

'''*************** 定义层添加函数 ****************'''
def add_layer(inputs, in_d, out_d, activation_function=None):
    #定义一个添加层函数
    #inputs 输入数据
    #in_size 输入数据阶数
    #out_size 输出数据阶数
    #n_layer
    #activation_function激励函数
    weights = tf.Variable(tf.random_normal([in_d, out_d])) #指定W的名称
    biases = tf.Variable(tf.zeros([1, out_d])+0.1)
    wx_plus_b = tf.matmul(inputs, weights)+biases 
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

keep_prob=tf.placeholder(tf.float32)

def myNet():
    xs=tf.placeholder(tf.float32, [None, d])
    ys=tf.placeholder(tf.float32, [None, d])
    weights = tf.Variable(tf.random_normal([d, d])) #指定W的名称
    biases = tf.Variable(tf.zeros([1, d])+0.1)
    wx_plus_b = tf.matmul(xs, weights)+biases 
    #y_model=weights*xs+biases
    return wx_plus_b,xs,ys,weights,biases

def myNet2():
    xs=tf.placeholder(tf.float32, [None, d])
    ys=tf.placeholder(tf.float32, [None, d])
    l1=add_layer(xs, d, 10, activation_function=tf.nn.relu)
    l2=add_layer(l1, 10, d, activation_function=None)
    return l2,xs,ys


def kerasNet():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(d,1)),
        keras.layers.Dense(500,activation=tf.nn.relu),
        keras.layers.Dense(100,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

#prediction,xs,ys,w,b=myNet()
#prediction,xs,ys=myNet2()
model=kerasNet()

def learn1():
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    '''*************** 计算 ****************'''
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            loss_run=sess.run(loss, feed_dict={xs: x_data, ys:y_data })
            #w_run=sess.run(w)
            #b_run=sess.run(b)
            print(loss_run)
            print('------------------')

def learn2(model,x_data,y_data):
    model.fit(x_data, y_data, epochs=15)
    #model.save_weights('my_checkpoint')    #手动保存权重
    return model

learn2(model, x_data, y_data)
