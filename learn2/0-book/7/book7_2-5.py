# coding=utf-8

from keras.utils import plot_model
from keras import layers,models
from keras.optimizers import RMSprop



max_features = 10000
maxlen = 500

def getModel():
	model = models.Sequential()
	model.add(layers.Embedding(max_features, 128, input_length=maxlen, name='embed'))   # 嵌入层,序列向量字典(10000,128)
	model.add(layers.Conv1D(32, 7, activation='relu'))  # 添加一个1D卷积层，卷积窗口长度7，32个特征
	model.add(layers.MaxPooling1D(5))                   # 添加一个1D池化层，池化窗口长度5
	model.add(layers.Conv1D(32, 7, activation='relu')) 
	model.add(layers.GlobalMaxPooling1D())              # 
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'] )

	plot_model(model, show_shapes=True, to_file='./images/model_7_2-5.png')	# 将模型表示为层组成的图

if __name__ == '__main__':
	getModel()
