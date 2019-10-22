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

img_path = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test4.jpg'
#加载保存模型
model=models.load_model('cats_and_dogs_small_5.2.5.h5')

# 输入数据
img = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
x=image.img_to_array(img) # 将其转换为形状(150,150,3)的numpy数组
x=x.reshape((1,) + x.shape) # 将其转换为形状(1,150,150,3)

# print(x.shape)
predictions = model.predict(x)
print(predictions)
p=predictions[0][0]
if p>=0.5:
    p=1
else:
    p=0
print(cla[p])

