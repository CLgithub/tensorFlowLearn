#coding=utf-8

# 显示几个随机增强后的训练图像

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing import image

train_cats_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small/train/cats'
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir) ]

img_path = fnames[0]	#选择一张图像进行增强

img = image.load_img(img_path, target_size=(150,150))	# 读取图片并调整大小
x=image.img_to_array(img) # 将其转换为形状(150,150,3)的numpy数组
x=x.reshape((1,) + x.shape) # 将其转换为形状(1,150,150,3)

# 构建数据生成器
datagen = ImageDataGenerator(
	rotation_range=40,	# 角度值(在 0~180 范围内)，表示图像随机旋转的角度范围
	width_shift_range=0.2, # 图像在水平或垂直方向上平移的范围(相对于总宽度或总高度的比例)
	height_shift_range=0.2, 
	shear_range=0.2, # 随机错切变换的角度
	zoom_range=0.2, # 图像随机缩放的范围
	horizontal_flip=True, # 随机将一半图像水平翻转。如果没有水平不对称的假设(比如真实世界的图像)，这种做法是有意义的
	fill_mode='nearest' #是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移。 我们来看一下增强后的图像
)

i=0
for batch in datagen.flow(x, batch_size=1):
	plt.figure(i)
	imgplot=plt.imshow(image.array_to_img(batch[0]))
	i += 1
	if i%5 ==0:
		break
plt.show()


