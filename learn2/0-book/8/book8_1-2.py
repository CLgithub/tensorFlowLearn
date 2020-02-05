# coding=utf-8

# 使用尼采的一些作品进行学习，生成尼采的写作风格和主题的文本序列

import keras
from keras import layers, models, Input
import numpy as np
import sys
import random


maxlen = 60 # 每个序列有60个字符
step = 3 	# 没3个字符采样一个新序列

def getData():
	# path = keras.utils.get_file('./data/nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
	path='./data/book6_4-3.py'
	text = open(path).read().lower()
	# print(len(text))

	sentences = [] # 保存所提取的序列
	next_chars = [] # 保存目标
	for i in range(0, len(text)-maxlen, step):	# 从0开始到len(text)-maxlen，没step个得到一个i
		sentences.append(text[i: i+maxlen])
		next_chars.append(text[i+maxlen])
	# print(len(sentences))

	chars = sorted(list(set(text)))	# 去重复排序得到 字符库
	char_indices = dict((char, chars.index(char)) for char in chars)	# 得到字符索引字典

	#  对字符进行one-hot编码
	x = np.zeros( (len(sentences),maxlen,len(chars)), dtype='int')	# 输入x的形状 (序列个数, 每个序列长度, 字符量)
	y = np.zeros( (len(sentences),len(chars)), dtype='int')			# 输出y的形状 (序列个数, 字符量)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1

	return x, y, char_indices, text, chars

def getModel(char_indices):
	model = models.Sequential()

	model.add(layers.LSTM( 128, input_shape=(maxlen, len(char_indices)) ))
	# model.add(layers.Conv1D(32, 7, input_shape=(maxlen, len(char_indices)), activation='relu'))
	# model.add(layers.MaxPooling1D(5))
	# model.add(layers.Conv1D(32, 7, activation='relu')) 
	# model.add(layers.GlobalMaxPooling1D())

	model.add(layers.Dense(len(char_indices), activation='softmax' ))
	# model.summary()

	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.01))	# one-hot编码，loss=categorical_crossentropy
	return model

# 给定模型预测，采样下一个字符的函数
def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds)/temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def createText(x, y, model, text, char_indices, chars):
	for epoch in range(1,60):	# 将模型训练60轮
		print()
		print('轮数：',epoch)
		model.fit(x, y, batch_size=128, epochs=1)
		start_index = random.randint(0, len(text)-maxlen -1)	# 随机选取一个开始点
		generated_text = text[start_index: start_index+maxlen]
		print('======"'+generated_text+'"======')

		for temperature in [0.2, 0.5, 1.0, 1.2]:	# 常识一些列不同的温度
			print()
			print('-----------------------'*2,temperature)
			print()
			sys.stdout.write(generated_text)

			for i in range(400):	# 从种子文本开始，生成400个字符
				sampled = np.zeros( (1, maxlen, len(char_indices)) )	# 对目前的生成的字符进行one-hot编码
				for t, char in enumerate(generated_text):
					sampled[0, t, char_indices[char]] = 1

				preds = model.predict(sampled, verbose=0)[0]
				next_index = sample(preds, temperature)
				next_char = chars[next_index]

				generated_text += next_char
				generated_text = generated_text[1:]

				sys.stdout.write(next_char)


if __name__ == '__main__':
	x, y, char_indices, text, chars=getData()
	model=getModel(char_indices)
	createText(x, y, model, text, char_indices, chars)


