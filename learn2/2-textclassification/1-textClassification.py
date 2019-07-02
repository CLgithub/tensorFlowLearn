#coding=utf-8

'''
https://www.tensorflow.org/tutorials/keras/basic_text_classification
影评文本分类
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.datasets import imdb

#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

np_load_old = np.load # save np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # modify the default parameters of np.load
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # call load_data with allow_pickle implicitly set to true
np.load = np_load_old # restore np.load for future normal usage
print(train_data[0])
