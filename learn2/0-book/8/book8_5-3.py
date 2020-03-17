# coding = utf-8

from keras import layers, Input
from keras.models import Model
from keras import backend as K
import keras
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

n = 2
latent_dim = 32  # 潜在空间维度
height = 32*n
width = 32*n
channels = 3  # 通道数

batch_size = 10

# 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)


# 生成器
def getGenerator():
	generator_input = Input(shape=(latent_dim,))
	# 将输入转换为大小为 16×16 的128个通道的特征图
	x = layers.Dense(128 * 16*n * 16*n)(generator_input)
	x = layers.LeakyReLU()(x)
	x = layers.Reshape((16*n, 16*n, 128))(x)  # (16,16,128)

	x = layers.Conv2D(256, 5, padding='same')(x)  # (16,16,256)
	x = layers.LeakyReLU()(x)

	# 反向卷积，上采样
	x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # (32,32,256)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(256, 5, padding='same')(x)  # (32,32,256)
	x = layers.LeakyReLU()(x)
	x = layers.Conv2D(256, 5, padding='same')(x)  # (32,32,256)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)  # (32,32,channels)
	generator = Model(generator_input, x)
	generator.summary()
	return generator


# 判别器
def getDiscriminator():
	discriminator_input = layers.Input(shape=(height, width, channels))
	x = layers.Conv2D(128, 3)(discriminator_input)
	x = layers.LeakyReLU()(x)
	x = layers.Conv2D(128, 4, strides=2)(x)
	x = layers.LeakyReLU()(x)
	x = layers.Conv2D(128, 4, strides=2)(x)
	x = layers.LeakyReLU()(x)
	x = layers.Conv2D(128, 4, strides=2)(x)
	x = layers.LeakyReLU()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.4)(x)  # 添加dropout，很重要
	x = layers.Dense(1, activation='sigmoid')(x)

	discriminator = Model(discriminator_input, x)

	discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
	discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, )

	discriminator.summary()
	return discriminator


train_dir = './data/cats_and_dogs_small/train/'
# validation_dir = './data/cats_and_dogs_small/validation/'

def cdate():
	#数据预处理
	train_datagen=ImageDataGenerator( rescale=1./255)  #设置放缩比例
	train_generator=train_datagen.flow_from_directory(  #构建python生成器,是一个类似迭代器的对象,从目录中读取图像数据并预处理
		train_dir,  #目标目录
		target_size=(width, height), #将所有图片的大小调整为150*150
		batch_size=batch_size,          #生成器每批次样本数量
		# class_mode='binary'     #因为使用了binary_crossentropy损失，所以需要用二进制标签
	)
	return train_generator



# 先用<真假图片--标签>训练判别器，冻结住判别器，再用<空间点--标签(谎言都是真的)>来训练gan（此时只能训练到生成器朝着真实方向改变）
def gan():
	generator = getGenerator()
	discriminator = getDiscriminator()
	discriminator.trainable = False  # 设置生成器为不可训练

	gen_input = Input(shape=(latent_dim,))
	gan_output = discriminator(generator(gen_input))  # 组合成gan
	gan = Model(gen_input, gan_output)

	gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
	gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
	gan.summary()

	# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
	# # print(y_train.flatten()==6) # [ True False False ... False False False]
	# x_train = x_train[y_train.flatten() == 6]  # y_train(50000,1)展平，得到[6 9 9 ... 9 1 1]，取出为6的标签所在的index，即取出所有标签为6的图片


	iteration = 10000
	save_dir = './data/gan/'

	start = 0
	for step in range(iteration):
		train_generator =cdate()
		real_images = next(train_generator)[0]

		random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))  # 从潜在空间中采样随机点  采样形状(20,32)
		generated_images = generator.predict(random_latent_vectors)  # 将这些点解码为虚假图片, (20,32,32,3)

		# real_images = x_train[start: start + batch_size]
		combined_images = np.concatenate([generated_images, real_images])

		labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])  # 生成的图片标签是1，真实的图片标签是0
		labels += 0.05 * np.random.random(labels.shape)  # 向标签中添加噪声

		d_loss = discriminator.train_on_batch(combined_images, labels)  # 训练判别器,有了初步的判别标准

		random_latent_vectors = np.random.normal(size=(batch_size, latent_dim)) # 重新采样20个点
		misleading_targets = np.zeros((batch_size, 1))  # 合并标签，全部是真实图像，（这是在故意撒谎）
		a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)  # 训练gan,此时判别器是冻结住的，所有全部为真的反馈会作用于生成器，让生成器去朝着生成更真实的方向去改变自己

		# start +=batch_size
		# if start > len(x_train)-batch_size:
		# 	start = 0

		if step%100 ==0:    # 没100步
			gan.save_weights('gan.h5')  # 保存权重
			print('discriminator loss:', d_loss)
			print('adversarial loss:', a_loss)

			img = image.array_to_img(generated_images[0]*255., scale=False)     # scale 评级
			img.save(os.path.join(save_dir, 'generated_frog'+str(step)+'.png'))    # 保存生成图片
			img = image.array_to_img(real_images[0]*255., scale=False)
			img.save(os.path.join(save_dir, 'real_frog'+str(step)+'.png'))     # 保存真实图片用于对比


# getGenerator()
# getDiscriminator()
gan()

# train_generator =cdate()
# x_train = next(train_generator)[0].shape
# print(x_train)
