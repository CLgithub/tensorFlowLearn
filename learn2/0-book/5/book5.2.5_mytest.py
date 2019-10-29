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

cla=['🐱','🐶']
cla1=['猫','狗']

img_paths=[]
img_path1 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test17.jpg'
#img_path2 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test15.jpg'
#img_path3 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test16.jpg'
img_paths.append(img_path1)
#img_paths.append(img_path2)
#img_paths.append(img_path3)

xs=[]
for img_path in img_paths:  #将图片转换成array
    img1 = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
    xs.append(x1)

x=np.array(xs)

#x=x.reshape((1,) + x.shape) # 将其转换为形状(1,150,150,3)

#加载保存模型
model=models.load_model('cats_and_dogs_small_5.2.5.h5')
predictions = model.predict(x)
print(predictions)
for p in predictions:
    if p[0]>=0.5:
        cod=1
    else:
        cod=0
    print(cla1[cod],end='')
    print(cla[cod],end='\t')

