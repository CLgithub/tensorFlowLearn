# coding = utf-8

from keras import layers, Input, Model
import numpy as np


# w1 = 37  # 第一层宽度
c_size = 5  # 卷积窗口大小
s_long = 2  # 步长

def test():

	c1 = valid_c(82)
	c2 = same_c(c1)
	c3 = valid_ct(c2)
	c4 = same_ct(c3)

	print(c1,c2,c3,c4)

	input = Input(shape=(82, 82, 3))
	y = layers.Conv2D(16, c_size, strides=s_long, activation='relu', )(input)
	y = layers.Conv2D(16, c_size, strides=s_long, activation='relu', padding='same')(y)

	y = layers.Conv2DTranspose(32, c_size, strides=s_long, activation='relu')(y)
	y = layers.Conv2DTranspose(32, c_size, strides=s_long, activation='relu', padding='same')(y)
	model = Model(input, y)
	model.summary()

# padding默认 valid     卷积:     w2=(w1-c_size)/s_long+1 反卷积:    w1=s_long(w2-1)+c_size
# padding='same' 卷积:    w2=w1/s_long+w1%s_long    反卷积:    w1 =w2(s_long+s_long*s_long)/(s_long+1)
def valid_c(w1):
	return int((w1-c_size)/s_long+1)
def valid_ct(w2):
	return int(s_long*(w2-1)+c_size)
def same_c(w1):
	return int(w1/s_long+w1%s_long)
def same_ct(w2):
	return int(w2*(s_long+s_long*s_long)/(s_long+1))


test()
