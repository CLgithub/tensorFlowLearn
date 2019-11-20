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
img_path11 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test11.jpg'
img_path12 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test12.jpg'
img_path13 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test13.jpg'
img_path14 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test14.jpg'
img_path15 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test15.jpg'
img_path16 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test16.jpg'
img_path17 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test17.jpg'
img_path18 = '/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/my_test/my_test18.jpg'
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
img_paths.append(img_path11)
img_paths.append(img_path12)
img_paths.append(img_path13)
img_paths.append(img_path14)
img_paths.append(img_path15)
img_paths.append(img_path16)
img_paths.append(img_path17)
img_paths.append(img_path18)
xs=[]
for img_path in img_paths:  #将图片转换成array
    img1 = image.load_img(img_path, target_size=(150,150))   # 读取图片并调整大小
    x1=image.img_to_array(img1) # 将其转换为形状(150,150,3)的numpy数组
    xs.append(x1)
x=np.array(xs)

#x=x.reshape((1,) + x.shape) # 将其转换为形状(1,150,150,3)

# 检测
# model=models.load_model('cats_and_dogs_small_5.2.5.h5') #加载保存模型
model=models.load_model('cats_and_dogs_small_5.3.2.h5') #加载保存模型
predictions = model.predict(x)
print(predictions)
for p in predictions:
    if p[0]>=0.5:
        cod=1
    else:
        cod=0
    print(cla1[cod],end='')
    print(cla[cod],end='\t')

