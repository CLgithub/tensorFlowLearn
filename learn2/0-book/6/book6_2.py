# coding=utf-8

# 在IMDB数据上使用Embedding层和分类器

from keras.datasets import imdb
from keras import preprocessing
from keras import models
from keras.layers import Flatten, Dense, Embedding
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

allWordLen=10000		# 作为特征的单词个数
sWordLen=20			# 在这么多单词后，截断文本

# 1.将数据加载为整数列表
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=allWordLen) 
# print(x_train[0])
# print(y_train[0])

# 2.将整数列表转换成形状为(len(samples), sWordLen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=sWordLen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=sWordLen)
# print(x_train.shape)	# Embedding层输入 (25000,20)

# 3.
model = models.Sequential()
# 指定Embedding层的最大输入长度，以便后面将嵌入输入展平，Embedding层激活(输出)的形状为(samples,sWordLen,8) (25000,20,8)
model.add(Embedding(10000, 8, input_length=sWordLen))	# 将整数索引映射为密集向量
model.add(Flatten()) # 将三维的嵌入张量展平成形状为(samples, sWordLen*8)的二维张量
model.add(Dense(1, activation='sigmoid')) # 添加分类器


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# for layer in model.layers:
# 	if layer.name=='embedding_1':
# 		print(layer.output.shape)	# Embedding层输出 (25000,20,8)
print(model.layers[0].input.shape)
print(model.layers[0].weights)
print(model.layers[0].output.shape)
model.summary()

# history=model.fit(x_train, y_train,
# 	epochs=10,
# 	batch_size=32,
# 	validation_split=0.2
# 	)

def show2(t_loss,t_acc,v_loss,v_acc):
    epochs=range(1, len(t_loss)+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, t_loss, 'b', label='t_loss')
    plt.plot(epochs, v_loss, 'r', label='v_loss')
    plt.ylim([0,1])
    plt.title('loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, t_acc, 'b', label='t_acc')
    plt.plot(epochs, v_acc, 'r', label='v_acc')
    plt.ylim([0,1])
    plt.title('acc')
    plt.legend()
    plt.show()

# t_loss=history.history['loss']
# t_acc=history.history['acc']
# v_loss=history.history['val_loss']
# v_acc=history.history['val_acc']
# show2(t_loss,t_acc,v_loss,v_acc)