#coding=utf-8

'''
总结图片与数组相互转换的几种方法
'''
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


myarray=[[0,0,0,255,255,255],
         [0,0,0,255,255,255],
         [0,0,0,255,255,255],
         [255,255,255,0,0,0],
         [255,255,255,0,0,0],
         [255,255,255,0,0,0]
         ]
npArray=np.array(myarray)
print(npArray.shape)
print('-------------------')
'''**************** opencv array>img *******************'''
cv2.imwrite('./image2/0-opencv.jpg',npArray)

'''**************** opencv img>array *******************'''
array1=cv2.imread('./image2/0-opencv.jpg') #读取图片,默认是带rgb的
print(array1.shape)
#img=cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE) #读取图片,灰度图形式
gray_array1=cv2.cvtColor(array1, cv2.COLOR_RGB2GRAY)
print(gray_array1.shape)

print('-------------------')
'''**************** PIL array>img *******************'''
img2=Image.fromarray(npArray.astype(np.uint8))
img2.save('./image2/1-pil.jpg')
'''**************** PIL img>array *******************'''
array2=np.asarray(Image.open('./image2/1-pil.jpg'))
#print(array2)
print(array2.shape)

print('-------------------')
'''**************** tensorflow array>img *******************'''
'''**************** tensorflow img>array *******************'''
with tf.Session() as sess:
    tensor1=tf.read_file('./image2/1-pil.jpg')
    tensor1_data=tf.image.decode_jpeg(tensor1,channels=3)
    print(tensor1_data.shape)
