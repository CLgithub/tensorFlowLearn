#coding=utf-8

# 多输出模型，例如输入某个匿名人士的一系列社交媒体发帖，然后尝试预测那个人的属性，比如年龄、性别和收入水平

from keras import models, layers, Input
import numpy as np
import keras

vocabulary_size = 50000		# 定义帖子单词库数量
age_size=100
income_size=5	# 5种类型收入
num_income_groups = 10


def getModel_api():
	posts_input = Input(shape=(None,), dtype='int32', name='posts')
	# embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)	# 感觉写反了
	embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)

	x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)	# 128个特征 窗口长度5
	x = layers.MaxPooling1D(5)(x)
	x = layers.Conv1D(256, 5, activation='relu')(x)
	x = layers.Conv1D(256, 5, activation='relu')(x)
	# x = layers.MaxPooling1D(5)(x)		# 加上后会报错：Computed output size would be negative:
	x = layers.Conv1D(256, 5, activation='relu')(x)
	x = layers.Conv1D(256, 5, activation='relu')(x)
	x = layers.GlobalMaxPooling1D()(x)
	x = layers.Dense(128, activation='relu')(x)

	age_prediction = layers.Dense(1, name='age')(x)	# 年龄输出 输出层都具有名称
	income_prediction = layers.Dense(10, activation='sigmoid', name='income')(x)	# 定义收入输出
	gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)	# 性别输出

	model = models.Model(posts_input, [age_prediction, income_prediction, gender_prediction])	# 年龄-回归 收入-多分类单标签 性别-二分类

	# 多输出模型的编译选项：多重损失,	你可以在编译时使用损失组成的列表或 字典来为不同输出指定不同损失
	model.compile(loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], optimizer='rmsprop', loss_weights=[0.25, 1.0, 10.0]) 	# 平衡损失
	# model.compile(loss=['age':'mse', 'income':'categorical_crossentropy', 
	#	'gender':'binary_crossentropy'], optimizer='rmsprop', loss_weights=['age':0.25, 'income':1.0, 'gender':10.0])
	return model


num_samples = 10
max_length = 100

def run(model):
	posts = np.random.randint(1, vocabulary_size, size=(num_samples, max_length))
	age_targets = np.random.randint(1, age_size, size=(num_samples))
	income_predictions = np.random.randint(1,income_size, size=(num_samples,num_income_groups))
	gender_predictions = np.random.randint(2, size=(num_samples))

	# print(age_targets)

	# model.fit(posts, [age_targets, income_predictions, gender_predictions], epochs=10, batch_size=64)
	model.fit(posts, {'age':age_targets, 'income':income_predictions, 'gender':gender_predictions}, epochs=10, batch_size=64)




if __name__ == '__main__':
	model=getModel_api()
	# model.summary()
	run(model)