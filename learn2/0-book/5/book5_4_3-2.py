#coding=utf-8

# 卷积神经网络可视化：可视化类激活的热力图，应用于自己的模型

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
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2

# 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)


cla=['🐱','🐶']
cla1=['猫','狗']

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

img_path = './data/my_test/my_test5.jpg'

img_paths=[]
img_paths.append(img_path)
xs=[]
for img_path in img_paths:  #将图片转换成array
    img1 = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
    x1 /= 255   # 数据预处理
    xs.append(x1)
x=np.array(xs)
# x = preprocess_input(x)

# print(x.shape)

def getHeatMap(x):
	# model = VGG16(weights='imagenet')
	model=models.load_model('cats_and_dogs_small_5.2.5.h5')
	# model.summary()
	preds=model.predict(x)
	print(preds[0][0])

	classNo=0
	for p in preds:
	    if p[0]>=0.5:
	        classNo=1
	    else:
	        classNo=0
	print(cla[classNo])

	# print(model.output)


	african_elephant_output = model.output[:, 0]	# 模型输出的，对应预测的类别输出公式
	
	last_conv_layer = model.get_layer('conv2d_4') # 最后一个卷积层

	# 得到 预测类别 在 block5_conv3层输出特征图 的剃度，即 block5_conv3层输出特征图 对 预测类别 的影响
	grads = backend.gradients(african_elephant_output, last_conv_layer.output)[0]
	pooled_grads = backend.mean(grads, axis=(0, 1, 2))	# 形状为 (512,) 的向量，每个元素是特定特征图通道的梯度平均大小

	# 得到 剃度值和损失值 与输入图像 的关系
	iterate = backend.function([model.input], [last_conv_layer.output[0], pooled_grads])

	last_conv_layer_value, pooled_grads_value = iterate([x])
	for i in range(128):	# 512个通道
		last_conv_layer_value[:, :, i] *= pooled_grads_value[i]	# 将特征图的每个通道 * 这个通道 对于 类别 的 影响程度

	heatmap = np.mean(last_conv_layer_value, axis=-1)	# 得到的特征图的逐通道平均值 即 类激活热力图
	return heatmap


def showHeatMap(heatmap):
	# 显示热力图	标准化到0～1范围，方便可视化
	heatmap = np.maximum(heatmap, 0)	# 去掉大于0的
	heatmap /= np.max(heatmap)		# 自己／自己最大的
	plt.matshow(heatmap)
	plt.show()
	return heatmap

def showHeatMapImg(heatmap):
	img=cv2.imread(img_path)	# 用cv2加载原始图像
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0] ))	# 将热力图的形状大小调整为与原始图像相同
	heatmap = np.uint8(255 * heatmap)	# 转换成RGB格式
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)	# 将热力图应用于原始图像
	superimposed_img = heatmap*0.4 +img# 这里的 0.4 是热力图强度因子
	cv2.imwrite('/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/heat.jpg', superimposed_img)


def func1():
	heatmap = getHeatMap(x)
	heatmap=showHeatMap(heatmap)
	showHeatMapImg(heatmap)

func1()

