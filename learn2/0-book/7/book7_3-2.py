#coding=utf-8

# 猫🐱 狗🐶 图片分类器，基础模型，深度可分离二维卷积神经网络搭建模型，超参数自动调节工具hyperas

import os, shutil
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import hyperas
from hyperas.distributions import choice
from hyperas import optim

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

original_dataset_dir='../5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
original_dataset_dir='../5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
base_dir='../5/data/cats_and_dogs_small'  #保存较小数据集的目录
base_dir='../5/data/cats_and_dogs_small'  #保存较小数据集的目录
#os.mkdir(base_dir)
train_dir=os.path.join(base_dir, 'train')   #训练
validation_dir=os.path.join(base_dir, 'validation') #校验
test_dir=os.path.join(base_dir, 'test') #测试
train_cats_dir=os.path.join(train_dir, 'cats')
train_dogs_dir=os.path.join(train_dir, 'dogs')
validation_cats_dir=os.path.join(validation_dir, 'cats')
validation_dogs_dir=os.path.join(validation_dir, 'dogs')
test_cats_dir=os.path.join(test_dir, 'cats')
test_dogs_dir=os.path.join(test_dir, 'dogs')

# 准备数据
def copyData():
    #将前1000张猫的图片复制到train_cats_dir
    fnames=['cat.{}.jpg'.format(i) for i in range(1000)]  #{}是占位符，填写format(i)中的i 得到cat.0.jpg,cat.1.jpg...
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    #将500张猫的图片复制到validation_cats_dir
    fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    #将500张猫的图片复制到test_cats_dir
    fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    #将500张猫的图片复制到validation_cats_dir
    fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    #将500张猫的图片复制到test_cats_dir
    fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

#copyData()

def getData():
	#数据预处理
	train_datagen=ImageDataGenerator(rescale=1./255)    #设置放缩比例
	test_datagen=ImageDataGenerator(rescale=1./255)

	train_generator=train_datagen.flow_from_directory(  #构建python生成器,是一个类似迭代器的对象,从目录中读取图像数据并预处理
	    train_dir,  #目标目录
	    target_size=(150, 150), #将所有图片的大小调整为150*150
	    batch_size=20,          #生成器每批次样本数量
	    class_mode='binary'     #因为使用了binary_crossentropy损失，所以需要用二进制标签
	    )
	validation_generator=test_datagen.flow_from_directory(
	    validation_dir,
	    target_size=(150, 150),
	    batch_size=20,
	    class_mode='binary'
	    )
	return train_generator,validation_generator

# 普通二维卷积神经网络搭建模型
def getModel_Conv2D(train_generator, validation_generator):
	model=models.Sequential()
	model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Conv2D( 64, (3,3), activation='relu' ))	# choice自动选择其一
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Conv2D(128, (3,3), activation='relu' ))
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Conv2D(128, (3,3), activation='relu' ))
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Flatten())
	model.add(layers.Dense({{choice([32, 64, 128, 512])}}, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	# model.summary()
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'] )


    

	history=model.fit_generator(
	    train_generator,
	    steps_per_epoch=100,
	    epochs=5,
	    validation_data=validation_generator,
	    validation_steps=50,
	    # callbacks= callbacks
	    )


	model.save('cats_and_dogs_small_7_3-2.h5')  #保存模型


	score, acc = model.evaluate(next(validation_generator), verbose=0)

	# return history
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def show2(t_loss,t_acc,v_loss,v_acc):
    epochs=range(1, len(t_loss)+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, t_loss, 'b', label='t_loss')
    plt.plot(epochs, v_loss, 'r', label='v_loss')
    plt.ylim([0,1])
    plt.title('loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, t_acc, 'b', label='t_acc')
    plt.plot(epochs, v_acc, 'r', label='v_acc')
    plt.ylim([0,1])
    plt.title('acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
	train_generator,validation_generator = getData()
	# getModel_Conv2D(train_generator,validation_generator)

	optim.minimize(model=getModel_Conv2D)

	# t_loss=history.history['loss']
	# t_acc=history.history['acc']
	# v_loss=history.history['val_loss']
	# v_acc=history.history['val_acc']

	# show2(t_loss,t_acc,v_loss,v_acc)
