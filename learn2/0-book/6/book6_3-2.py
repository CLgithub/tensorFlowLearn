# coding=utf-8

# 基准方法、密集连接

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras import models, layers, optimizers


data_dir = './data'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

def getFloatData():
	f = open(fname)
	data = f.read()
	f.close()

	lines = data.split('\n')
	header=lines[0].split(',')
	lines=lines[1:]

	float_data = np.zeros((len(lines), len(header)-1))

	for i, line in enumerate(lines):
		values = [float(x) for x in line.split(',')[1:]]
		float_data[i, :]=values

	mean = float_data[:200000].mean(axis=0)
	float_data -= mean
	std = float_data[:200000].std(axis=0)
	float_data /= std
	return float_data

# temp = float_data[:, 1]
# plt.plot(range( 1440 ), temp[:1440])
# plt.show()


# 从原始数据的min_index到max_index之间，根据设置，取出样本数据是过去lookback个时间步，目标是delay后的数据的温度的 <输入-输出>组合
# data:浮点数数据组成的原始数组
# lookback：输入数据应该包括过去多少个时间步
# delay：目标应该在未来多少个时间步之后
# min_index 和 max_index:data 数组中的索引，用于界定需要抽取哪些时间步。这有
	# 助于保存一部分数据用于验证、另一部分用于测试
# shuffle:是打乱样本，还是按顺序抽取样本
# batch_size:每个批量的样本数
# step:数据采样的周期(单位:时间步)。我们将其设为 6，为的是每小时抽取一个数据点
def generator(data, lookback, delay, min_index, max_index, 
	shuffle=False, batch_size=128, step=6):

	if max_index is None:
		max_index = len(data) - delay -1
	i = min_index + lookback 	# 从哪里开始取,
	while 1:
		if shuffle:
			rows = np.random.randint(i, max_index, size=batch_size)	#从i ~ max_index，随机取batch_size个
		else:
			if i+batch_size > max_index-1:	# i+batch_size得到应该要取到的索引 > 最大索引
				i = min_index+lookback 		# 回到最开始能取的
			rows = np.arange(i, i+batch_size)	# 取哪些
			i += len(rows)

		# print(rows) # 得到可以取哪些数据的索引序列，rows得到索引，取哪些p元素的索引
		samples = np.zeros((batch_size, lookback//step, data.shape[-1])) # (一次取多少,多少个时间点,每个数据的形状)
		targets = np.zeros((batch_size, )) # 具体温度数据
		# 具体怎么取
		for j, row in enumerate(rows):
			indices = range(row-lookback, row, step)	# 从输入数据开始，到输入数据结束的这些行取数据，没step行取一次
			samples[j] = data[indices]
			targets[j] = data[row + delay][1]	# 有了第row行的数据，目标是delay步后的数据

		yield samples, targets


lookback = 1440	# 6*24*10 10天 输入数据应该包括过去
step = 6	# 
delay = 144		# 目标应该在未来多少个时间步之后 1天
batch_size = 128	# 每个批量的样本数
def getData(float_data):
	train_gen = generator(data=float_data, lookback=lookback, delay=delay, 
		min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)

	val_gen = generator(data=float_data, lookback=lookback, delay=delay, 
		min_index=200001, max_index=300000, step=step, batch_size=batch_size)

	test_gen = generator(data=float_data, lookback=lookback, delay=delay, 
		min_index=300001, max_index=None, step=step, batch_size=batch_size)

	val_steps = (300000 - 200001 - lookback) // batch_size	# 查看整个验证集，需要从val_gen中抽取多少次
	test_steps = (len(float_data) - 300001 - lookback) // batch_size	# 

	return train_gen, val_gen, test_gen, val_steps, test_steps


# 基于常识的预测：24小时后的温度等于现在的温度，使用平均绝对误差(MAE)指标来评估这种方法
def evaluate_naive_method(val_gen, val_steps):
	batch_maes = []
	for step in range(val_steps):
		samples, targets = next(val_gen)	
		preds = samples[:, -1, 1]
		mae = np.mean(np.abs(preds - targets))
		batch_maes.append(mae)
	print(np.mean(batch_maes))	
	# MAE=0.29，温度数据被标准化成均值为0、标准差为1，所以无法直接对这个值进行解释。
	# 它转化成温度的平均绝对误差为 0.29×temperature_std 摄氏度，即 2.57°C。std[1]是均方差  MAE=(温度-平均值)/均方差

# 密集连接
def dense_meatod(float_data, train_gen, val_gen, val_steps):
	model = models.Sequential()
	model.add(layers.Flatten(input_shape=( lookback//step, float_data.shape[-1] )))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(1))

	model.compile(loss='mae', optimizer=optimizers.RMSprop())

	history = model.fit_generator(
		train_gen,	# 数据生产器
		steps_per_epoch=500,	# 每轮抽取多少批次的生成器的数据，每批次128，共200000，
		epochs=20,				# 训练轮次
		validation_data=val_gen,		# 验证集，可以是numpy数组组成的元祖，也可以是数据生成器
		validation_steps=val_steps 		# 从验证集中抽取多少个批次用于评估
		)
	return history

def show2(t_loss,v_loss):
    epochs=range(1, len(t_loss)+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, t_loss, 'b', label='t_loss')
    plt.plot(epochs, v_loss, 'r', label='v_loss')
    plt.ylim([0.2,0.4])
    plt.title('loss')
    plt.legend()
    # plt.subplot(1,2,2)
    # plt.plot(epochs, t_acc, 'b', label='t_acc')
    # plt.plot(epochs, v_acc, 'r', label='v_acc')
    # plt.ylim([0,1])
    # plt.title('acc')
    # plt.legend()
    plt.show()


if __name__ == '__main__':

	float_data=getFloatData()
	train_gen, val_gen, test_gen, val_steps, test_steps = getData(float_data)

	evaluate_naive_method(val_gen, val_steps)
	history = dense_meatod(float_data, train_gen, val_gen, val_steps)
	t_loss=history.history['loss']
	# t_acc=history.history['acc']
	v_loss=history.history['val_loss']
	# v_acc=history.history['val_acc']
	show2(t_loss,v_loss)


