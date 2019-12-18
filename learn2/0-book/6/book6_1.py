# coding=utf-8

import numpy as np
import string
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']


# 自定义单词级oneHot编码
def oneHotW():
	token_index = {}	# 构建数据中所有标记的索引
	for sample in samples:
		for word in sample.split():	# sample.split() 按单词分割
			if word not in token_index:
				token_index[word] = len(token_index)	# 刚好可以用原来的长度作为新的索引 'The':0

	print(token_index)	# 获取到数据中所有标记的索引

	max_length = 12 # 定义每个样例的长度，不足的补0，超过的不要	
	results = np.zeros( shape=(len(samples), max_length, len(token_index)) )	#(样本数, 每个样本采样长度, 采集到的标记数)
	# 这样处理的好处是每个样本得到的张量都是一样的

	results_2=np.zeros( (len(samples), len(token_index)) )
	for i, sample in enumerate(samples):
		for j, word in list(enumerate(sample.split()))[:max_length]:
			index = token_index.get(word) # 
			results[i, j, index]=1.
			results_2[i,index]=1

	# print(results)
	print(results_2)

# 自定义字符级oneHot编码
def oneHotChar():
	token_index = {}	# 构建数据中所有标记的索引
	characters = string.printable	# 所有可打印的ASCII字符
	token_index = dict( zip(range(0, len(characters)), characters) )	# 获取到数据中所有标记的索引

	max_length = 50 # 定义每个样例的长度，不足的补0，超过的不要	
	results = np.zeros( shape=(len(samples), max_length, len(token_index)) )
	for i, sample in enumerate(samples):
		for j, char in enumerate(sample):
			index=token_index.get(char)
			results[i, j, index]=1
	print(results)

# 用Keras实现单词级oneHot编码
def oneHotW_keras():
	tokenizer = Tokenizer(num_words=10)	# 创建一个分词器(tokenizer)，设置为只考虑前1000个最常见的单词
	tokenizer.fit_on_texts(samples) 	# 用分词器对 样本 构建单词索引
	
	print(tokenizer.word_index) # 获取字典
	
	one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')	# one-hot 关联 方式
	print(one_hot_results)	# 获取转换后的数据
	
	# sequences = tokenizer.texts_to_sequences(samples) 	# 将字符串转换为整数索引组成的列表
	# print(sequences)

# one-hot编码的一种变体是所谓的one-hot散列技巧(one-hot hashing trick)，如果词表中唯 一标记的数量太大而无法直接处理，就可以使用这种技巧
def oneHotW_hashingTrick():
	dimensionality=1000 # 将单词保存为长度为 1000 的向量。如果单词数量接近 1000 个(或更多)那么会遇到很多散列冲突，这会降低这种编码方法的准确性
	max_length = 10 

	results_2=np.zeros( (len(samples), dimensionality) )
	results = np.zeros((len(samples), max_length, dimensionality))
	for i ,sample in enumerate(samples):
		for j, word in list(enumerate(sample.split()))[:max_length]:
			# print(abs(hash(word)) )
			# word的hash码的绝对值 对 dimensionality取余，所以要dimensionality远大于需要散列的唯一标记的个数，散列冲突的可能行很小
			index=abs(hash(word)) % dimensionality	# 将单词散列为 0~dimensionality 范围内的一个随机整数索引
			results[i, j, index]=1
			results_2[i, index]=1

	print(results)
	# print(results_2)


# oneHotW()
# oneHotChar()
oneHotW_keras()
# oneHotW_hashingTrick()


