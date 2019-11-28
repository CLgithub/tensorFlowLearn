#coding=utf-8

# 猫🐱 狗🐶 图片分类器，实践测试

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import VGG16 # 导入VGG16模型
# from book5_3_1 import extract_features

cla=['🐱','🐶']
cla1=['猫','狗']


# 准备数据
img_paths=[]
img_path1 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test1.jpg'
img_path2 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test2.jpg'
img_path3 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test3.jpg'
img_path4 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test4.jpg'
img_path5 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test5.jpg'
img_path6 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test6.jpg'
img_path7 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test7.jpg'
img_path8 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test8.jpg'
img_path9 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test9.jpg'
img_path10 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test10.jpg'
img_paths.append(img_path1)
img_paths.append(img_path2)
img_paths.append(img_path3)
img_paths.append(img_path4)
img_paths.append(img_path5)
img_paths.append(img_path6)
img_paths.append(img_path7)
img_paths.append(img_path8)
img_paths.append(img_path9)
img_paths.append(img_path10)
xs=[]
for img_path in img_paths:  #将图片转换成array
    img1 = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
    x1 /= 255   # 数据预处理
    xs.append(x1)
x=np.array(xs)

# 将数据输入到conv_base中
conv_base = VGG16(  # 构建卷积基
        weights='imagenet', # 指定模型初始化的权重检查点
        include_top=False,  # 指定模型最后是否包含密集连接分类器
        input_shape=(150,150,3) # 输入到网络中的图像张量的形状（可选），如果不传，网络可以处理任意形状的输入
        )
test_features = conv_base.predict(x)
print(test_features.shape)

# 转换形状，便于输入到模型中
test_features=np.reshape(test_features, (len(xs),4*4*512))
print(test_features.shape)

# 将卷积基的输出输入到模型中
model=models.load_model('cats_and_dogs_small_5.3.1.h5') # 加载保存模型
predictions = model.predict(test_features)
print(predictions)
for p in predictions:
    if p[0]>=0.5:
        cod=1
    else:
        cod=0
    print(cla1[cod],end='')
    print(cla[cod],end='\t')

