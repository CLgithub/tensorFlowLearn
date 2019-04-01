#coding=utf-8
'''
基础学习：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-3-session/
'''
import tensorflow as tf

'''
1. session会话控制 Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
'''
m1 = tf.constant([[3,3]])   #constant常数,一个二维的一行两列(1,2)的张量
m2 = tf.constant([[2],[2]]) #constant常数，一个二维的2行1列(2,1)的张量
product = tf.matmul(m1, m2) #m1*m2 此时并没有正在的开始计算

#方法1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

'''
#方法2
with tf.Session() as sess:  #类似匿名方式
    result2 = sess.run(product)
    print(result2)
'''

print(' ----------------------------------------------------------------------')
'''
2.Variable 变量
'''
state = tf.Variable(0, name='counter')  #定义一个变量
one = tf.constant(1) #定义一个常量,
new_value = tf.add(state, one) #定义加法步骤 (注: 此步并没有直接计算)
update = tf.assign(state, new_value) #将 State 更新成 new_value
#初始化
#如果定义 Variable, 就一定要 initialize
#init = tf.initialize_all_veriables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

print(' ----------------------------------------------------------------------')
'''
3.占位符
placeholder 是 Tensorflow 中的占位符，暂时储存变量.
'''
i1 = tf.placeholder(tf.float32) #定义两个占位符
i2 = tf.placeholder(tf.float32)

#o = tf.multiply(i1, i2) #普通乘法
o = tf.matmul(i1,i2) #矩阵乘法

with tf.Session() as sess:
    print(sess.run(o, feed_dict={i1:[[7,2]],i2:[[2],[3]]}))


