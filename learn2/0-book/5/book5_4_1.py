#coding=utf-8

# 卷积神经网络可视化：可视化卷积神经网络的中间输出（中间激活），
# 自我观察：猫的最后一个池化层会比较繁杂，狗的相反

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.applications import VGG16 # 导入VGG16模型
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

original_dataset_dir='/home/ubuntu/develop/tensorFlowLearn/learn2/0-book/5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
original_dataset_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
base_dir='/home/ubuntu/develop/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small'  #保存较小数据集的目录
base_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small'  #保存较小数据集的目录
#os.mkdir(base_dir)
train_dir=os.path.join(base_dir, 'train')   #训练
validation_dir=os.path.join(base_dir, 'validation') #校验
test_dir=os.path.join(base_dir, 'test') #测试
train_cats_dir=os.path.join(train_dir, 'cats')
train_dogs_dir=os.path.join(train_dir, 'dogs')
validation_cats_dir=os.path.join(validation_dir, 'cats')
validation_dogs_dir=os.path.join(validation_dir, 'dogs')
test_cats_dir=os.path.join(test_dir, 'cats')
test_dogs_dir=os.path.join(test_dir, 'dogs')

test_dir = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small/test'
# 准备数据
def getTestImg():
	img_paths=[]
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test1.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test2.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test3.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test4.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test5.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test6.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test7.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test8.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test9.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test10.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test11.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test12.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test13.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test14.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test15.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test16.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test17.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test18.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test19.jpg')
	# img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test20.jpg')
	img_paths.append('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/Figure_1.png')
	xs=[]
	for img_path in img_paths:  #将图片转换成array
	    img1 = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
	    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
	    x1 /= 255   # 数据预处理
	    xs.append(x1)
	x=np.array(xs)
	return x

def showActiv(model,activation,layerNum):
	layer_names=[]
	for layer in model.layers[:layerNum]:
		layer_names.append(layer.name)

	images_per_row = 16	# 设置一排显示多少个

	for layer_name, layer_activation in zip(layer_names, activation):	# zip 将layer_names和activation打包成元组，然后遍历
		n_features = layer_activation.shape[-1]	# 不同层特征图的特征(通道)个数
		size = layer_activation.shape[1] # 不同层特征图的形状(图片个数,宽,高,n_features )，宽高相同，取宽

		n_cols = n_features // images_per_row	# 得到一列显示多少
		display_grid = np.zeros((size * n_cols, size * images_per_row))	# 得到一层应该有多少横竖(高,宽)
		# print('display_grid',display_grid.shape)

		for col in range(n_cols):
			for row in range(images_per_row):
				channel_image = layer_activation[0, :, :, col*images_per_row+row]	# 显示这一层的哪个通道
				channel_image -= channel_image.mean()
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')

				display_grid[ col*size:(col+1)*size, row*size:(row+1)*size ] = channel_image 	# display_grid的什么位置显示这个图片

		# plt.figure(figsize=(display_grid.shape[1]/size, display_grid.shape[0]/size))	# 指定一层宽高显示多少个(总宽/单个宽度,总高/单个高度)
		plt.figure(figsize=(images_per_row, n_cols))	# 指定单个层显示的宽和高(一列显示多少个,一列显示多少个)
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect='auto', cmap='viridis')

	plt.show()	



def func1():
	# 加载模型
	model = models.load_model('cats_and_dogs_small_5.2.h5')
	model.summary()

	testX=getTestImg()
	print(testX.shape)

	# 显示模型
	# plt.imshow(testX[0])
	# plt.show()

	layerNum=8
	layer_outputs = [layer.output for layer in model.layers[:layerNum]]	# 提取前8层的输出
	# 用一个输入张量，一个输出张量，实例化一个模型activation_model，该模型有一个输入，8个输出，即每个激活层一个输出
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
	activation=activation_model.predict(testX)	# 将测试数据给激活层模型，得到对应的激活
	
	# activation 长度为8，分别对应8层的激活
	print(activation[0].shape)

	# plt.matshow(activation[0][0,:,:,4],cmap='viridis')
	# plt.imshow(activation[0][0,:,:,4])

	# for layActivation in activation:
	# 	plt.matshow(layActivation[0,:,:,0],cmap='viridis')
	# plt.show()
	showActiv(model,activation,8)

	# a=32 / 16
	# print(a)

func1()











