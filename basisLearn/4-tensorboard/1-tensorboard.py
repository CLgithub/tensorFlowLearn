#coding=utf-8
#数据可视化，详情见：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


'''*************** 定义层添加函数 ****************'''
def add_layer(inputs, in_d, out_d, n_layer, activation_function=None):
    #定义一个添加层函数
    #inputs 输入数据
    #in_size 输入数据阶数
    #out_size 输出数据阶数
    #n_layer
    #activation_function激励函数
    layer_name = 'layer_%s'%n_layer  #laryer名称
    with  tf.name_scope(layer_name):   #指定布局名称为layer
        with tf.name_scope('Ws'):  #指定W的名称
            weights = tf.Variable(tf.random_normal([in_d, out_d]), name='W') #指定W的名称
            tf.summary.histogram(layer_name+'-w', weights) #设置记录weights的变化
        with tf.name_scope('bs'):
            biases = tf.Variable(tf.zeros([1, out_d])+0.1, name='b')
            tf.summary.histogram(layer_name+'-b', biases)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights)+biases 
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
            tf.summary.histogram(layer_name+'-outputs', outputs)
        return outputs

'''*************** 导入数据 ****************'''
d=2
t=d*100
x_data = np.linspace(-1, 1, t, dtype=np.float32).reshape(100,d) #在-1～1之间取100个点，然后变成100行1列的矩阵
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise    #square 计算各元素的平方
# y=x^2-0.5+noise

'''*************** 构建神经网络 d,10,d ****************'''
with tf.name_scope('inputs'):   #定义输入层名称
    xs = tf.placeholder(tf.float32, [None, d], name='x_input')     #给输入层元素定义名称
    ys = tf.placeholder(tf.float32, [None, d], name='y_input')
l1 = add_layer(xs, d, 10, n_layer=1, activation_function=tf.nn.relu) #定义l1层，将d阶变成10阶,设置该成名称为n1
prediction = add_layer(l1, 10, d, n_layer=2, activation_function=None)

'''*************** 计算损伤，然后反向传播并用随机梯地下降获取最优解 ****************'''
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1] )) #计算损伤
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  #随机梯度下降

'''*************** 初始化 ****************'''
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

'''*************** 显示拟合情况 ****************'''
fig = plt.figure()  #创建一个图像实例
ax = fig.add_subplot(1,1,1)     #设置布局
ax.scatter(x_data, y_data)
plt.ion() #代开交互模式

'''*************** 计算 ****************'''
ilist = []
losslist = []
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            for l in lines:
                ax.lines.remove(l)  #删除原来的线
        except Exception:
            pass
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, i)
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=1)
        plt.pause(0.1)
        loss_run=sess.run(loss, feed_dict={xs: x_data, ys:y_data })
        print(loss_run)

plt.show()

# 运行后会在logs目录下产生日志，通过命令‘tensorboard --logdir logs’来查看图形结构
