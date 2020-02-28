# coding=utf-8

from keras import backend as K
import numpy as np
import tensorflow as tf

# 返回一个随机正态分布值的抽样张量
t1 = K.random_normal(
		shape=((9,)),		# 抽取的形状
		mean=1.,			# 抽样的正态分布平均值
		stddev=1.,			# 抽样的正态分布的标准差
	)

# print(t1.shape)
# print(t1[1][1])

sess = tf.Session()
n1=sess.run(t1)

print(n1)

print(np.mean(n1))	# 均值
# print(np.var(n1))	# 方差
# print(np.std(n1))	# 标准差

print(n1[5])