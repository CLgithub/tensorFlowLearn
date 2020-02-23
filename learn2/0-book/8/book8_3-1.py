# coding=utf-8

from keras.preprocessing import image
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
import imageio

s_path='./data/style2/'
originnal_img_path=s_path+'content.png'	# 原始图像路径
style_reference_img_path = s_path+'style.png'		# 风格参考图像路径

width, height = image.load_img(originnal_img_path).size
img_height = 400
img_width = int(img_height/height * width)	# 等比放缩


# 定义内容损失
def content_loss(base, combination):
	return K.sum(K.square(combination - base))

# 定义风格损失，内部关系，内积
def style_loss(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_height * img_width
	s_loss= K.sum(K.square(S - C)) / ( 4. * (channels**2)*(size**2) )
	return s_loss

# 得到内部结构
def gram_matrix(x):
	features = K.batch_flatten(				# 批量_压平	？压平前为什么要先变形
		K.permute_dimensions(x, (2, 0, 1))	# 排列_尺寸,变形
	)
	gram = K.dot(features, K.transpose(features))	# 转制后，与原来相乘，得到内积
	return gram

# 它对生成 的组合图像的像素进行操作。它促使生成图像具有空间连续性，从而避免结果过度像素化
# 得到相邻像素的差别
def total_variation_loss(x):
	a = K.square(									# 在竖直方向上，用下一像素的框 减去 上一像素的框 
		x[:, :img_height-1, :img_width-1, :] -
		x[:, 1:, 			:img_width-1, :]
	)
	b = K.square(									# 在水平方向上，用右一像素的框 减去 左一像素的框 
		x[:, :img_height-1, :img_width-1, :] -
		x[:, :img_height-1, 1:,			  :]
	)
	# a+b相当于下上两个像素的差别+右左两个像素的差别，这些差别 做 元素级的指数运算操作，得到相邻像素的差别，如果这个差别小，就到达了抗锯齿的效果
	return K.sum(K.pow(a+b, 1.25))

# 加载图像
def preprocess_image(image_path):
	img = image.load_img(image_path, target_size=(img_height, img_width))
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	img_array = vgg19.preprocess_input(img_array)
	return img_array


# 将一个张量转换为有效图像,不直接使用array_to_img的原因在于，
# 图片经过了vgg19.preprocess_input()，作用是减去ImageNet的平均像素值，使其中心为0，这里相对于vgg19.preprocess_input 的逆操作
def deprocess_image(x):
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = x[:, :, ::-1]	# 将BGR，转位RGB，也是逆操作的一部分，图片长宽通道不变，RGB通道逆序
	x = np.clip(x, 0, 255).astype('uint8')
	return x


# 构建一个网络，输入：三张图片，输出：vgg19的激活，定义损失，使损失最小，反向调整输入，一遍学习是调整网络
# 定义损失函数，loss= distance
# 运行梯度下降
def getFetch_loss_and_grads():
	# 1 构建模型
	originnal_img = K.constant(preprocess_image(originnal_img_path))	#常数
	style_reference_img = K.constant(preprocess_image(style_reference_img_path))
	generated_img = K.placeholder((1, img_height, img_width, 3))	# 占位符

	input_tensor = K.concatenate([originnal_img, style_reference_img, generated_img], axis=0)	# 将三张图像合并为一个批量

	model = vgg19.VGG19(
		input_tensor = input_tensor,	# 利用三张图像组成的批量作为输入 来构建 VGG19 网络
		weights = 'imagenet',			# 加载模型将 使用预训练的 ImageNet 权重
		include_top = False				# 不加载顶部
	)


	# 2 构建损失函数
	outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])	# 得到字典 <layer_name, layer_out>
	content_layer = 'block5_conv2'	# 用于内容损失的层
	style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']	# 用于风格损失的层
	total_variation_weight = 1e-4	# 渐变系数
	style_weight = 1.
	content_weight = 0.025

	loss = K.variable(0.)
	layer_features = outputs_dict[content_layer]
	target_img_features = layer_features[0, :, :, :]	# 取第0个图像在内容层的激活		即 原始图像在该层的激活
	combination_features = layer_features[2, :, :, :]	# 取第2个图像在内容层激活		即 生成图像在该层的激活
	loss = loss + content_weight * content_loss(target_img_features, combination_features)	# 内容损失添加到总损失中

	for layer_name in style_layers:
		layer_features = outputs_dict[layer_name]
		style_img_features = layer_features[1, :, :, :]
		combination_features = layer_features[2, :, :, :]
		sl = style_weight*style_loss(style_img_features, combination_features)
		loss = loss + (style_weight/len(style_layers)) * sl 	# 每一层贡献一部分
	loss = loss + total_variation_weight * total_variation_loss(generated_img)

	# 3 构建梯度下降过程
	grads = K.gradients(loss, generated_img)[0]		# 得到损失相对于生成图片的梯度
	fetch_loss_and_grads = K.function([generated_img], [loss, grads])	# 构建函数，通过输入generated_img，得到[loss, grads]
	return fetch_loss_and_grads

# 由于SciPy只能将loss和梯度单独传入，
# 所以定义一个类，它可以同时计算损失值和梯度值，在第一次调用时会返回损失值，同时缓存梯度值用于下一次调用
class Evaluator():
	def __int__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		# assert self.loss_value is None	# assert断言，在表达式为Flase时触发异常
		x = x.reshape((1, img_height, img_width, 3))	# 图像变形
		fetch_loss_and_grads=getFetch_loss_and_grads()
		outs = fetch_loss_and_grads([x])
		loss_value = outs[0]
		grad_values = outs[1].flatten().astype('float64')
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


iterations = 20
evaluator = Evaluator()

def run():
	x = preprocess_image(target_img_path)
	x = x.flatten()		# 因为fmin_l_bfgs_b算法只能接收展品的x

	for i in range(iterations):
		print('Start of iteration', i)
		start_time = time.time()
		x, min_val, info = fmin_l_bfgs_b(	# l_bfgs 算法，scipy.optimize中的一种优化算法，最小化方法
			evaluator.loss,
			x,
			fprime=evaluator.grads,		# 接收梯度
			maxfun=20
		)
		img = x.copy().reshape((img_height, img_width, 3))
		img = deprocess_image(img)
		fname = s_path +'my_result_at_iteration_%d.png' % i
		imsave(fname, img)
		print('Image saved as', fname)
		end_time = time.time()
		print('Iteration %d completed in %ds' % (i, end_time - start_time))


# 拼凑图片
def addImg():
	img_width = 537
	img_height = 400
	img1 = image.load_img('/Users/l/Desktop/deepDream7.jpg', target_size=(img_height, img_width))
	img2 = image.load_img('/Users/l/Desktop/style.jpg', target_size=(img_height, img_width))
	img3 = image.load_img('/Users/l/Desktop/my_result_at_iteration_7.png', target_size=(img_height, img_width))

	img1 = image.img_to_array(img1)
	img2 = image.img_to_array(img2)
	img3 = image.img_to_array(img3)

	print(img1.shape)
	print(img2.shape)
	print(img3.shape)

	img_padd=20
	results = np.zeros((img_height + img_padd*2, img_width*3 + img_padd*4, 3))
	print(results.shape)

	results[img_padd:img_padd+img_height, img_padd*1+img_width*0:img_padd*1+img_width*1, :] = img1
	results[img_padd:img_padd+img_height, img_padd*2+img_width*1:img_padd*2+img_width*2, :] = img2
	results[img_padd:img_padd+img_height, img_padd*3+img_width*2:img_padd*3+img_width*3, ::] = img3
	
	results = results.astype('uint8')

	imageio.imwrite('/Users/l/Desktop/a.png', results)

# run()

addImg()







