#coding=utf-8
#手写数字识别

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

# 用占位符定义输入x
x = tf.placeholder("float", [None, 784])    #None表示可以输入任意张图片，28*28=784,None行，784列
# 用变量定义权重和偏移量
W = tf.Variable(tf.zeros([784,10])) #看图，x的竖方向上有784个，要得到10个数字的分别的可能性
b = tf.Variable(tf.zeros([10])) #每个结果加上偏移量
# y=Wx+b
y=tf.nn.softmax(tf.matmul(x,W)+b) #matmul执行乘法，不知这里为什么不是W,x，然后交给softmax计算各个数字的概率

# 设置训练方法，利用损失函数
y_ = tf.placeholder("float", [None,10]) #y_表示正确的概率
# 误差（双交熵）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    #计算正确答案与输出答案至今的差距
# 反向传播，修改w，b，使误差最小化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #gradient descent algorithm 随机梯度下降
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(-tf.reduce_sum( y_*tf.log(tf.nn.softmax(tf.matmul( x ,tf.Variable(tf.zeros([784,10])))+tf.Variable(tf.zeros([10]))))))

# 准备开始训练
init = tf.initialize_all_variables()    #先初始化
sess=tf.Session()
sess.run(init)
# 开始训练
for i in range(1):
    batch_xs, batch_ys=mnist.train.next_batch(100) #每次循环，随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #比较y与正确答案是否相等
#correct_prediction = tf.equal(y, y_) #比较y与正确答案是否相等
    #我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
