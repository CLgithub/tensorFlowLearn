#coding=utf-8

'''
https://www.tensorflow.org/tutorials/keras/basic_text_classification
影评文本分类
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb=keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)
