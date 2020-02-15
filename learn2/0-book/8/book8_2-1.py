# coding=utf-8

# 用keras实现deepDream

import keras
from keras import backend
from keras.applications import inception_v3
import numpy as np
import scipy
import imageio
from keras.preprocessing import image
import time

backend.set_learning_phase(0)		# 不需要训练模型，这个命令会禁止所有与训练有关的操作

def getFetch_loss_and_grads():
	model=inception_v3.InceptionV3(weights='imagenet', include_top=False)	# 指定模型初始化的权重检查点

	layer_contributions = {	# 配置一个字典，配置某层 对于要 最大化的损失 的贡献大小
		'mixed2': 0.2,
		'mixed3': 3,
		'mixed4': 2.,
		'mixed5': 1.5,
	}

	# model.summary()

	# 定义损失函数，损失就是mixed2,mixed3,mixed4,mixed5层激活的L2范数的加权求和，
	layer_dict = dict( [(layer.name, layer) for layer in model.layers] )	# 层字典 <层名称, 层>
	loss = backend.variable(0.)		# 实例化变量并返回它
	for layer_name in layer_contributions:
		coeff = layer_contributions[layer_name]	# 得到该成的贡献值
		activation = layer_dict[layer_name].output	# 得到该层的输出

		scaling = backend.prod(				# 在某一指定轴，计算张量中的值的乘积。
				backend.cast(						# 转换成float32类型
					backend.shape(activation), 	# 获取输出形状
					'float32'	#
				)
			)
		# 将该层特征的 L2 范数添加到 loss 中
		loss = loss + coeff * backend.sum(		# 求和*加权
				backend.square(				# 平方
					activation[:, 2:-2, 2:-2, :]	# shape(样本数, 宽度, 高度, 层数)
				)
			)/scaling



	# 得到了损失loss函数, 得到loss函数关于输入的梯度，利用梯度上升，使得loss最大化，
	dream = model.input	# 模型的输入，即目标
	grads = backend.gradients(loss, dream)[0]	# 得到loss相对于输入的梯度
	grads /= backend.maximum(			# 逐个元素比对两个张量的最大值
			backend.mean(				# 平均
				backend.abs(grads)	# 绝对值
			), 1e-7
		)
	# 那就需要一个方法：给定输入图像，它能够计算损失张量和梯度张量的值
	fetch_loss_and_grads = backend.function([dream], [loss, grads])	# 得到函数，根据输入，可以得到loss, grads
	return fetch_loss_and_grads


def eval_loss_and_grads(x):
	fetch_loss_and_grads=getFetch_loss_and_grads()
	outs = fetch_loss_and_grads([x])
	loss_value = outs[0]
	grads_values = outs[1]
	return loss_value, grads_values


# 利用梯度上升，使得loss最大化，此时的输入，就是目标
def gradient_ascent(x, iterations, step, max_loss=None):
	for i in range(iterations):
		localtime = time.asctime( time.localtime(time.time()))
		print(str(localtime), '变幻中。。。', i )
		loss_value, grads_values =eval_loss_and_grads(x)
		if max_loss is not None and loss_value>max_loss:
			break
		x +=step*grads_values
		a = step*grads_values
		print(a.shape,loss_value)
		save_img(x, fname='./data/dream/final_dream_'+str(i)+'.png')
	return x


step = 0.03 		# 梯度上升的步长
num_octave = 1 		# 运行梯度上升的尺度个数
octave_scale = 1	# 两个尺度之间的大小比例
iterations = 40 	# 每个尺度上运行梯度上升的步数

max_loss =20		# 如果损失增大到大于10，我们要中断梯度上升过程，以避免得到丑陋的伪影
base_image_path = './data/deepDream8.jpg'

def run():
	img_np = preprocess_image(base_image_path)		# 将基础图像加载成一个Numpy数组
	original_shape = img_np.shape[1:3]		# 获取图片的尺寸
	successive_shapes = [original_shape]
	
	for i in range(1, num_octave):
		shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])	# 将形状缩小为原来的octave_scale倍
		successive_shapes.append(shape)	# 添加到形状列表中，由大到小
	successive_shapes = successive_shapes[::-1]		# 倒序，由小到大

	original_img = np.copy(img_np)		# 保存原始图片
	shrunk_original_img = resize_img(img_np, successive_shapes[0])	# 原始图片变形为最小尺寸的图片

	for shape in successive_shapes:
		# print('--------------------------------',str(shape))
		# print(successive_shapes)
		img_np = resize_img(img_np, shape)	# 将梦境图片放大为当前
		img_np = gradient_ascent(img_np, iterations=iterations, step=step, max_loss=max_loss)	#运用梯度提升，进行变幻

		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)	# 由上个尺寸变形为当前尺寸，有损失
		same_size_original = resize_img(original_img, shape)					# 原始尺寸变形为当前尺寸，得到当前尺寸原本的样子
		lost_detail = same_size_original - upscaled_shrunk_original_img			# 当前尺寸原本的样子 - 上个尺寸放大为当前尺寸的样子 = 上个尺寸到当前尺寸放大过程中的损失
		img_np +=lost_detail													# 弥补损失

		shrunk_original_img = resize_img(original_img, shape)					# 存储当前尺寸原本的样子，给下个尺寸用
		save_img(img_np, fname='./data/dream/dream_at_scale_' + str(shape) + '.png')

	save_img(img_np, fname='./data/dream/final_dream.png')

def preprocess_image(image_path):
	img = image.load_img(image_path)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)	# 在img形状的第0个位置添加数据
	img = inception_v3.preprocess_input(img) # 类似归一化的函数
	return img

def resize_img(img, size):
	img = np.copy(img)
	factors = (1,
		float(size[0])/img.shape[1],
		float(size[1])/img.shape[2],
		1)
	return scipy.ndimage.zoom(img, factors, order=1)	# 上采样与下采样

# def save_img(img_np, fname):
# 	img = image.array_to_img(img_np[0])
# 	imageio.imwrite(fname, img)

def save_img(img_np, fname):
	pil_img = deprocess_image(np.copy(img_np))
	imageio.imwrite(fname, pil_img)

def deprocess_image(x):		# 通用函数，将一个张量转换为有效图像
	if backend.image_data_format() == 'channels_first':
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((x.shape[1], x.shape[2], 3)) 
	x /= 2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	return x


if __name__ == '__main__':
	run()
	# img_np=preprocess_image('./data/deepDream1.png')
	# # print(img_np[0])
	# save_img2(img_np[0],'a.png')

	



