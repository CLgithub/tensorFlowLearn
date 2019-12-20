# coding=utf-8

# 理解循环神经网络

import numpy as np



timesteps = 10 #输入序列的时间步数，相对于单个序列的长度
input_features = 2 #输入特征空间的维度
output_features = 3 #输出特征空间的维度

inputs = np.random.random((timesteps, input_features))	#随机初始化一个序列


state_t = np.zeros((output_features,)) # t时刻的状态

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

successive_outputs = []
for input_t in inputs:
	output_t = np.tanh( np.dot(W,input_t) + np.dot(U,state_t) + b )
	state_t = output_t 		# 最关键的一步，将上一步的输出迭代入下一步
	successive_outputs.append(output_t)

final_output_sequence = np.stack(successive_outputs, axis=0)
print(final_output_sequence.shape) # (单个序列的长度,输出特征空间的维度) (10,3)

print(np.array(successive_outputs)[-1])	# 最重要的是最后一步的输出信息，已经包含了整个序列的信息



