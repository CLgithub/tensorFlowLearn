#coding=utf-8

# 多输入模型，问答分类问题

from keras import models, layers, Input
import numpy as np
import keras

text_vocabulary_size = 10000		# 文本输入词库长度
question_vocabulary_size = 10000	# 问题输入词库长度
answer_vocabulary_size = 500		# 回答输出词库长度


def getModel_api():
	text_input = Input(shape=(None,), dtype='int32', name='text')	# 定义文本输入，一个长度可变的整数序列
	embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)	# 将文本输入嵌入到64维的向量中
	text_tensor = layers.LSTM(32)(embedded_text)	# 利用LSTM将向量编码为单个向量

	question_input = Input(shape=(None,), dtype='int32', name='question')	# 定义问题输入
	embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)	# 将问题输入嵌入到64维德向量中
	question_tensor = layers.LSTM(16)(embedded_question)	# 利用LSTM将向量编码为单个向量

	concatenated = layers.concatenate([text_tensor, question_tensor], axis=1)	# 将文本向量和问题向量连接在一起
	answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)	# 交给softmax分类器

	model = models.Model([text_input, question_input], answer)	# 在模型实例化是指出输入输出
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'] )
	return model

num_samples = 1000
max_length = 100

def run(model):
	texts = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))	# 从1～词库长度 中随机抽取数据组成数组，模拟出 文本输入
	questions = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))	# 从1～词库长度 中随机抽取数据组成数组，模拟出 问题输入
	answers = np.random.randint(answer_vocabulary_size, size=(num_samples))

	answers = keras.utils.to_categorical(answers, answer_vocabulary_size)	# 回答使用one-hot编码

	# model.fit([texts, questions], answers, epochs=10, batch_size=128)	# 第一种训练方式 使用输入组成的列表来拟合
	model.fit({'text':texts, 'question':questions}, answers, epochs=10, batch_size=128)	# 第二种训练方式 使用输入组成的字典来拟合


if __name__ == '__main__':
	model=getModel_api()
	# model.summary()
	run(model)

