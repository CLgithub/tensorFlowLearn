#coding=utf-8
'''
激励函数:
    0.什么是激励函数:一种非线性变换
    1.为什么要使用激励函数：实际问题往往都是非线性的，激励函数是非线性的，使用激励函数，可以使其个性更明显，并过滤掉杂音
    2.常用的激励函数有：sigmid,tanh,relu,leaky relu
'''
'''
搭建一个神经网络
'''
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def add_layer(inputs, in_d, out_d, activation_function=None):
    #定义一个添加层函数
    #inputs 输入数据
    #in_size 输入数据阶数
    #out_size 输出数据阶数
    #activation_function 激励函数
    weights = tf.Variable(tf.random_normal([in_d, out_d]))    #定义一个随机变量矩阵w，因为要使得inputs[_,in_d]*w[,]=output[_,out_d]，能相乘，input的列=w的行，所以w的行数=in_d，结果有out_d列，所以w的列数=out_d
    biases = tf.Variable(tf.zeros([1, out_d])+0.1)
    wx_plus_b = tf.matmul(inputs, weights)+biases #结果有inputs的行，weigiths的列 组成
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

d=1
#导入数据
#x_data = np.linspace(-1, 1, 30, dtype=np.float32)[:, np.newaxis] #-1~1之间取300个点，然后变成竖直方向上的矩阵
x_data = np.linspace(-1, 1, 100, dtype=np.float32).reshape(100,d)
#x_data = np.random.normal(-10, 1, (4,1)).astype(np.float32)
print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise #square 计算各元素的平方

#构建神经网络 d,10,d 2层，第一层将d阶变成10阶，第二层将10阶变成3阶
xs = tf.placeholder(tf.float32, [None, d])
ys = tf.placeholder(tf.float32, [None, d])
l1 = add_layer(xs, d, 10, activation_function=tf.nn.relu)   #定义l1层，将d阶变成10阶
prediction = add_layer(l1, 10, d, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1] )) #计算损失
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #随机梯度下降方向传播损失

#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#数据可视化
fig=plt.figure() #创建一个图像实例
ax = fig.add_subplot(2,1,1) #设置布局
bx = fig.add_subplot(2,1,2)
ax.scatter(x_data, y_data)
#ax.plot(x_data, y_data) #滑线
plt.ion() #打开交互模式

ilist=[]
losslist=[]
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data}) #训练
    if i % 50 ==0:
        try:
            for l in lines:
                ax.lines.remove(l)  #删除原来的线
        except Exception:
            pass
        loss_values=sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        ilist.append(i)
        losslist.append(loss_values)

        lines = ax.plot(x_data, prediction_value, 'r-', lw=1)
        #bx.scatter(ilist,losslist)
        bx.plot(ilist,losslist)
        plt.pause(0.1) #每帧动画时间
plt.show()
