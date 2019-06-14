#coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

'''************* 定义输入集 ***************'''
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None, 784])  #图片规格28*28
    ys=tf.placeholder(tf.float32, [None, 10])   #结果为0～9

'''************* 定义层添加函数 ****************'''
def add_layer(inputs, in_d, out_d, n_layer, activation_function=None):
    #定义一个添加层函数
    #inputs 输入数据
    #in_size 输入数据阶数
    #out_size 输出数据阶数
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

'''************* 构建神经网络 ***************'''
prediction=add_layer(xs, 784, 10, 'layer1', activation_function=tf.nn.softmax)

'''************* 计算损伤 ***************'''
with tf.name_scope('loss'):
    loss=tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

'''************* 反向传播 ***************'''
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

'''************* 初始化 ***************'''
sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/', sess.graph)
sess.run(tf.global_variables_initializer())

#计算准确率
def compute_accuracy(v_xs, v_ys):
    #global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


'''************* 训练 ***************'''
for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 ==0:
        acc=compute_accuracy(mnist.test.images, mnist.test.labels)
        loss_run=sess.run(loss, feed_dict={xs:batch_xs, ys:batch_ys})
        print(loss_run,acc)
        merged_run=sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys})
        writer.add_summary(merged_run, i)

