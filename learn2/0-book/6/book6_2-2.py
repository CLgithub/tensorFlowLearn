# coding=utf-8

# Keras 中的循环层

from keras import models,layers

model = models.Sequential()
model.add(layers.Embedding(10000, 32, input_length=20 ))	# 嵌入层，嵌入字典形状(10000,32) 单个序列的长度20
# 为了提高网络的表示能力，将多个循环层逐个堆叠有时也是很有用的。在这种情况下，你 需要让所有中间层都返回完整的输出序列
model.add(layers.SimpleRNN(32, return_sequences=True))	# 第一个循环层，完整输出(batch_size,20,32)，相对于对每一个序列，都看完整过程
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(32 ))						# 最后一个循环层，输出(batch_size,32)，相当于对每一个序列，都只看最后一步
model.summary()