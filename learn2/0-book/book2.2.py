#coding=utf-8

'''
Deep Learning with Python 2.2 神经网络的数据表示

张量（tensor）:数据的容器，几乎总是数值数据
    维度(dimension) 又叫做 轴(axis) 轴的个数也叫做阶(rank),矩阵是一个二维张量,numpy中可以用ndim属性得到
        0维度的张量---标量，（单独的某个数字）
        1维度的张量---向量  [12,21,3,4,16]
        2维度的张量---矩阵  [[12,21,3,4,16],
                            [12,21,3,4,16]]
        3维度的张量---立方体[[[12,21,3,4,16],
                              [12,21,3,4,16]],
                             [[12,21,3,4,16],
                              [12,21,3,4,16]],
                             [[12,21,3,4,16],
                              [12,21,3,4,16]]]
                    shape=(3,2,5)

    形状(shape)，这是一个整数元祖，表示张量沿每个轴的维度大小(元素个数),
        先看方括号确定元祖元素个数，然后从大的开始拆解填入

    数据类型(dtype),元素的数据类型


'''

#from keras.datasets import fashion_mnist as mnist
from keras.datasets import mnist as mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

#plt.imshow(train_images[4], cmap=plt.cm.binary)
plt.imshow(train_images[5])
#plt.show()

tensor3d=np.array([[[12,21,3,4,16],
                    [12,21,3,4,16]],
                   [[12,21,3,4,16],
                    [12,21,3,4,16]],
                   [[12,21,3,4,16],
                    [12,21,3,4,16]]]
)

print(tensor3d.shape)
print('------------------------------ 2.2.6   在numpy中操作张量')
'''------------------------------ 2.2.6   在numpy中操作张量
张量切片：选择张量的特定元素
给出了切片沿着每个张量轴的起始索引和结束索引
:等同于选择整个轴
'''
my_slice=train_images[10:100]
print(my_slice.shape)
#等同于以下
my_slice=train_images[10:100, :, :]
print(my_slice.shape)
my_slice=train_images[10:100, 0:28, 0:28]
print(my_slice.shape)
#选择所有图像的右下角14*14的区域
my_slice=train_images[:,14:,14:]
print(my_slice.shape)
plt.imshow(my_slice[3])
#plt.show()
#选择所有图像的中央14*14的区域
my_slice=train_images[:,7:-7,7:-7]
print(my_slice.shape)
plt.imshow(my_slice[3])
#plt.show()
print('------------------------------ 2.2.8   现实世界中的数据张量')
'''------------------------------ 2.2.8   现实世界中的数据张量
向量数据：2D张量(矩阵)，shape=(samples, features)   (样本,特征)
时间序列数据或时间数据：3D张量，shape=(samples, timesteps, features)(样本,时间序列,特征)
图像：4D张量,shape=(samples,height,width,channels)(tensorflow约定)，shape=(samples,channels,height,width)(theano约定)
    入mist数据，shape=(60000,28,28,1)灰度图，颜色通道为1,是否可以省略以后探究
视频：5D张量，shape=(samples,frames,height,width,channels)
    此处frames可理解为帧，时间维度
    多个视频，多个样本samples
'''
