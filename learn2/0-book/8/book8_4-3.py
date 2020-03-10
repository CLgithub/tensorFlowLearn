# coding = utf-8

from keras import Input, layers, models, metrics, backend as K
from keras.datasets import mnist
from keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.preprocessing.image import ImageDataGenerator

img_size = 80
img_shape = (img_size, img_size, 3)
batch_size = 16
latent_dim = 2 		# 潜在空间的维度，一个二维平面


train_dir='./data/cats_and_dogs_small/train/'   #训练
validation_dir='./data/cats_and_dogs_small/validation/'

# 利用卷积基进行数据预处理
def cdate():
    #数据预处理
    train_datagen=ImageDataGenerator(
        rescale=1./255,  #设置放缩比例
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest'
    )    
    test_datagen=ImageDataGenerator(rescale=1./255) #不能增强验证数据

    train_generator=train_datagen.flow_from_directory(  #构建python生成器,是一个类似迭代器的对象,从目录中读取图像数据并预处理
        train_dir,  #目标目录
        target_size=(img_size, img_size), #将所有图片的大小调整为150*150
        batch_size=200,          #生成器每批次样本数量
        class_mode='binary',     #因为使用了binary_crossentropy损失，所以需要用二进制标签
        # color_mode='grayscale'
        )
    validation_generator=test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=200,
        class_mode='binary',
        # color_mode='grayscale'
        )
    return train_generator,validation_generator

# 定义编码模型
def enCoder(input_img):
	x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
	x = layers.Conv2D(64, 3, activation='relu', padding='same', strides=(2,2))(x)
	x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
	x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

	shape_before_flattening = K.int_shape(x)	# 得到x的尺寸元祖	(14, 14, 64)

	x = layers.Flatten()(x)
	x = layers.Dense(32, activation='relu')(x)

	# 将x分向两个全连接分支，将这两个分支定义为z_mean z_log_var
	z_mean = layers.Dense(latent_dim)(x)	# 两个分支形状都是(2)
	z_log_var = layers.Dense(latent_dim)(x)

	# m_enCoder = models.Model(input_img, z_mean)
	# m_enCoder.summary()
	return z_mean, z_log_var, shape_before_flattening

# 潜在空间采样方法
def sampling(args):
	epsilon = K.random_normal(						# 返回正态分布值的张量
		shape=( K.shape(z_mean)[0], latent_dim ), 	# shape=(2, 2)
		mean=0., 									# 抽样的正态分布平均值
		stddev=1.									# 抽样的正态分布标准差
	)	#
	z = z_mean + K.exp(0.5*z_log_var)*epsilon		# 得到采样点
	return z

# 定义解码器层
def getDeCoder(z, shape_before_flattening):
	decoder_input = layers.Input(K.int_shape(z)[1:])	# 去掉样本数维度得到(2,)

	# 相当于展平的反操作
	out = layers.Dense(
		np.prod(shape_before_flattening[1:]), 	# 返回指定轴上元素的乘积，
		activation='relu'
	)(decoder_input)
	out = layers.Reshape(shape_before_flattening[1:])(out)	# 变形	(14, 14, 64)

	# 该层是转置的卷积操作(反卷进)，data_format='channels_last'，将x反卷积为(img_size,img_size,32)
	out = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(out) 	# (img_size, img_size, 32) 2x+1=img_size x=14
	out = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(out)	#(sampling, img_size, img_size, 1)

	m_deCoder = models.Model(decoder_input, out)
	return m_deCoder

# 自定义层，用于集合两个损失
class MyLayer(layers.Layer):
	def vae_loss(self, x, z_decoded):
		x = K.flatten(x)
		z_decoded = K.flatten(z_decoded)
		xent_loss = metrics.binary_crossentropy(x, z_decoded)	# 获取x与z_decoded之间的二进制交叉熵，重构损失
		kl_loss = -5e-4 * K.mean(								# 正则化损失
			1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), # 
			axis= -1
		)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):	# 通过编写一个 call 方法来实现自定义层
		x = inputs[0]
		z_decoded = inputs[1]
		loss = self.vae_loss(x, z_decoded)
		self.add_loss(loss, inputs=inputs)	# 为自定义层，创建损失，层必须要有损失
		return x 							# 该层有了损失，返回x，进行经过该层前后x变化的前后比较

def run(vea_model):
	# (x_train, _), (x_test, y_test) = mnist.load_data()
	# train_x = x_train.astype('float32') / 255.
	# train_x = train_x.reshape(train_x.shape + (1,))
	# test_x = x_test.astype('float32') / 255.
	# test_x = test_x.reshape(test_x.shape + (1,))
	# history=vea_model.fit(
	# 	x=train_x, y=None,
	# 	shuffle=True,
	# 	epochs=10,
	# 	batch_size=batch_size, 
	# 	validation_data=(test_x, None)
	# )
	# return history

	train_generator, validation_generator=cdate()
	train_x = next(train_generator)[0]
	test_x = next(validation_generator)[0]
	history=vea_model.fit(
		x=train_x, y=None,
		shuffle=True,
		epochs=200,
		batch_size=batch_size, 
		validation_data=(test_x, None)
	)
	return history

# 显示潜入空间
def displayVEA(m_deCoder):
	n = 10
	digit_size = img_size
	figure = np.zeros((digit_size * n, digit_size * n, 3))	# 总图片
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))	# 在指定的间隔内返回均匀间隔的数字,
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))	# norm.ppf()正态分布的累计分布函数的逆函数，即下分位点
	# grid_x = np.linspace(0.05, 0.95, n)
	# grid_y = np.linspace(0.05, 0.95, n)

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])	# (1,2)
			# z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)	# 把 z_sample 沿着xi方向复制batch_size倍得到(1,32)，变形(16,2)，为了取16个，其实只取一个也行
			# print(z_sample.shape)
			x_decoded = m_deCoder.predict(z_sample,)	# 解码得到图片 (batch_size, img_size, img_size, 1)
			# print(x_decoded.shape)
			digit = x_decoded[0].reshape(digit_size, digit_size, 3)	# 指取第0个解码出的图片
			# print(digit.shape)
			figure[i*digit_size: (i+1)*digit_size, j*digit_size: (j+1)*digit_size] = digit 	# 填入相应位置
	
	# image_path='./data/cats_and_dogs_small/validation/dogs/dog.1025.jpg'
	# digit = Image.open(image_path)
	# digit = digit.resize((img_size, img_size))
	# figure[1*digit_size: (1+1)*digit_size, 1*digit_size: (1+1)*digit_size] = digit 
	img1 = image.array_to_img(figure)
	img1.show()

def imgToImg(vea_model):
	image_path='./data/cats_and_dogs_small/validation/dogs/dog.1025.jpg'

	img = Image.open(image_path)
	# img = img.convert('L')
	# print(img.shape)
	img = img.resize((img_size, img_size))

	img_array = np.array(img)
	img_array = np.expand_dims(img_array, axis=0)
	# img_array = np.expand_dims(img_array, axis=3)

	i1 = vea_model.predict(img_array)
	i1 = i1[0]
	i1=np.array(i1).reshape(img_size,img_size,3)

	img1 = image.array_to_img(i1)
	img1.show()






if __name__ == '__main__':
	#--------------------- 创建VEA 开始 ----------------------#
	input_img = Input(shape=img_shape)
	# 编码层
	z_mean, z_log_var, shape_before_flattening = enCoder(input_img)
	z = layers.Lambda(sampling)([z_mean, z_log_var])	# 相当于直接调用z = sampling(z_mean, z_log_var), z就是潜在空间中的一个点
	
	# 解码层
	m_deCoder = getDeCoder(z, shape_before_flattening)
	# m_deCoder.summary()
	z_decoded = m_deCoder(z)	# 得到解码后的z

	# 损失组合层
	input_img1 = MyLayer()([input_img, z_decoded])

	# 构建模型
	vea_model = models.Model(input_img, input_img1)
	vea_model.compile(optimizer='rmsprop', loss=None)
	# vea_model.summary()

	# 训练
	history = run(vea_model)
	vea_model.save('model_8_4-2_vea.h5')
	m_deCoder.save('model_8_4-2_deCoder.h5')
	#--------------------- 创建VEA 结束 ----------------------#

	#--------------------- 潜在空间到图片 开始 ----------------------#
	# m_deCoder=models.load_model('model_8_4-2_deCoder.h5')
	displayVEA(m_deCoder)
	#--------------------- 潜在空间到图片 结束 ----------------------#

	#--------------------- 图片到图片 开始 ----------------------#
	# vea_model=models.load_model('model_8_4-2_vea.h5')
	# vea_model.summary()
	# imgToImg(vea_model)
	#--------------------- 图片到图片 结束 ----------------------#







