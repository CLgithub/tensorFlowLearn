#coding=utf-8

# 卷积神经网络可视化：可视化过滤器（显示每个过滤器响应的视觉模式）

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras import backend
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


def deprocess_image(x):
	# 对张量做标准化，使其均值为0，标准差为0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	# 将x裁切(clip)到[0,1]区间
	x += 0.5
	x = np.clip(x, 0, 1)
	# 将x转换为RGB数组
	x *= 225
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def generate_pattern(layer_name, filter_index, input_img_data, size=150):
	model = VGG16(  # 构建卷积基
		weights='imagenet', # 指定模型初始化的权重检查点
		include_top=False,  # 指定模型最后是否包含密集连接分类器
		input_shape=(size,size,3)
		)
	# model = models.load_model('cats_and_dogs_small_5.2.h5')

	# model.summary()

	layer_output = model.get_layer(layer_name).output 	#取出该模型的block3_conv1层的输出，形状(图片数量, 宽, 高, 通道数)
	loss = backend.mean(layer_output[:, :, :, filter_index])		#取出输出的第0个过滤器，求其在某一指定轴的均值，作为损失函数，

	# 为了实现梯度下降，我们需要得到损失相对于模型输入的梯度，使用gradients函数
	grads = backend.gradients(loss, model.input)[0] 	#gradients返回的是一个张量列表（本例中列表长度为1）。因此，只保留1个
	# 为了让梯度下降过程顺利进行，一个非显而易见的技巧是将梯度张量除以其 L2 范数(张量中所有值的平方的平均值的平方根)来标准化。这就确保了输入图像的更新大小始终位于相同的范围。
	grads /= (backend.sqrt(backend.mean(backend.square(grads))) + 1e-5)	#grads/=(grads的平方，求评价，再开方+1e-5),做除法前加上1e-5,以防不小心除以0
	# print(layer_output)
	# print(loss)
	# print(grads)

	# 现在你需要一种方法:给定输入图像，它能够计算损失张量和梯度张量的值
	iterate = backend.function([model.input], [loss, grads])	# iterate是一个函数，它将一个Numpy张量(model.input)转换为两个Numpy张量(loss,grads)组成的列表
	loss_value, grads_value = iterate([np.zeros((1, size, size, 3))])	# zeros空白输入开始

	
	step = 1.
	for i in range(40):	# 运行40次
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step	# 沿着让损伤最大化的方向调节输入图像

	img = input_img_data[0]
	return deprocess_image(img)

def func1():
	input_img_data = np.random.random((1, size, size, 3)) * 20 +128.	# 从一张带有噪声的灰度图开始
	plt.imshow(generate_pattern('block3_conv1', input_img_data, 0 ))
	plt.show()

def getImageData(img_path, size):
	img_paths=[]
	xs=[]
	img_paths.append(img_path)
	for img_path in img_paths:  #将图片转换成array
	    img1 = image.load_img(img_path, target_size=(size,size))   # 读取图片并调整大小
	    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
	    x1 /= 255   # 数据预处理
	    xs.append(x1)
	x=np.array(xs)

	print(x.shape)
	return x


def func2():
	layer_name = 'block4_conv1'
	size = 64
	margin = 5

	input_img_data = np.random.random((1, size, size, 3)) * 20 +128.	# 从一张带有噪声的灰度图开始
	# input_img_data = getImageData('./data/my_test11.png', size)

	results = np.zeros((8*size+7*margin, 8*size+7*margin, 3))	# 空图像，用于保存结果

	for r in range(8):	# 行
		for c in range(8):	# 列
			filter_img = generate_pattern(layer_name, c+(r*8), input_img_data, size=size )	# 行列依次投入进去得到响应
			print(c+(r*8)+1)

			r_start = r*size + r*margin		# 得到该图片行的开始位置
			r_end = r_start + size 			# 得到该图片行的结束位置
			c_start = c*size + c*margin		# 得到该图片列的开始位置
			c_end = c_start + size
			results[r_start:r_end, c_start:c_end, :]=filter_img
	results=results/255
	plt.figure(figsize=(20, 20))
	plt.title(layer_name)
	plt.imshow(results)
	plt.show()


# func1()
func2()
'''
a0 = np.zeros((1,2,3))
a=np.array([[[95, 92, 9,]]])
a0[0:1, 0:1, :]=a
a0=a0/255
plt.imshow(a0)
plt.show()
'''






