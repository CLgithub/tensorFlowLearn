# coding=utf-8

# 用keras实现deepDream

import keras
from keras import backend
from keras.applications import inception_v3
import numpy as np
import scipy
import imageio
from keras.preprocessing import image


def resize_img(img, size):
	img = np.copy(img)
	factors = (1,float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
	return scipy.ndimage.zoom(img, factors, order=1)

def preprocess_image(image_path):
	img = image.load_img(image_path)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img) 
	return img

# lost_detail = resize_img(original_img, shape) - resize_img(resize_img(original_img, shape), shape)


original_img = preprocess_image('./data/deepDream2.jpg')

a=resize_img(original_img, shape)
b=resize_img(resize_img(original_img, shape), shape)

print(a)
print(b)