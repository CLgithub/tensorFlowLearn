#coding=utf-8

# 猫🐱 狗🐶 图片分类器，使用预训练的卷积神经网络：使用VGG16进行模型微调

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.applications import VGG16 # 导入VGG16模型
import numpy as np
import tensorflow as tf

# 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

original_dataset_dir='/home/ubuntu/develop/tensorFlowLearn/learn2/0-book/5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
original_dataset_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
base_dir='/home/ubuntu/develop/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small'  #保存较小数据集的目录
base_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/cats_and_dogs_small'  #保存较小数据集的目录
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


# 利用卷积基进行数据预处理
def cdate():
    #数据预处理
    train_datagen=ImageDataGenerator(
        rescale=1./255,  #设置放缩比例
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )    
    test_datagen=ImageDataGenerator(rescale=1./255) #不能增强验证数据

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

# 构建模型
def getModel():
    conv_base = VGG16(  # 构建卷积基
        weights='imagenet', # 指定模型初始化的权重检查点
        include_top=False,  # 指定模型最后是否包含密集连接分类器
        input_shape=(150,150,3) # 输入到网络中的图像张量的形状（可选），如果不传，网络可以处理任意形状的输入
        )

    #搭建模型，只需要密集连接层
    model=models.Sequential()
    model.add(conv_base)    # 构建模型，直接添加卷积基
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False # 设置卷积基不可训练，此设置针对model模型中来说，对conv_base本身无影响

    # 编译模型
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc']
            )

    return model,conv_base

# 设置模型的各层是否可训练
def setLayerIsTra(model,conv_base):    
    conv_base.trainable = True # 在model层，设置卷积基层可训练
    
    # 对卷积基内部block5_conv1及其以后的层进行解冻
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    for layer in model.layers:
        print(layer.name,layer.trainable)
    print(len(model.trainable_weights))
    # model.summary()

    # 再次编译
    model.compile(loss='binary_crossentropy', 
            optimizer=optimizers.RMSprop(lr=1e-5),  #降低学习率，是希望变化范围不要太大
            metrics=['acc']
        )
    return model,conv_base

def run(model, train_generator, validation_generator):
    # 训练
    history=model.fit_generator(    #开始训练，fit_generator在数据生成器上的效果和fit相同
        train_generator,      #数据生成器,可以不停的生成输入和目标组成的批量
        steps_per_epoch=100,    # 每一轮抽取多少批次的生成器生成的数据，本例中，每批量20，共2000，所以每轮抽取100个批次数据生成器的数据，轮训完一轮用完所有图片
        epochs=30,              # 轮训次数
        validation_data=validation_generator,   #验证集，可以是numpy数组组成的元祖，也可以是数据生成器
        validation_steps=50                 # 从验证集中抽取多少个批次用于评估
        )

    # 训练后，保存前，看看卷积基是否能被训练，便于模型微调使用
    # for layer in model.layers:
    #     print(layer.name,layer.trainable)
    # print(len(model.trainable_weights))
    # model.summary()

    # 保存训练结果
    model.save('cats_and_dogs_small_5.3.3.h5')  #保存模型
    return history

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


def func1():
    # # 1.重新训练
    # # copyData()
    train_generator, validation_generator=cdate()
    # model,conv_base=getModel()  # 搭建模型
    # # history0=run(model, train_generator, validation_generator)  # 训练添加层

    # 2.加载5_3_2的模型
    model=models.load_model('cats_and_dogs_small_5.3.2.h5')
    conv_base = ''
    # print(len(model.trainable_weights))
    for layer in model.layers:
        # print(layer.name,layer.trainable)
        if layer.name == 'vgg16':
            conv_base = layer
    # print(conv_base.summary())


    model,conv_base=setLayerIsTra(model,conv_base)  # 微调模型
    history=run(model, train_generator, validation_generator)  # 再次训练
    t_loss=history.history['loss']
    t_acc=history.history['acc']
    v_loss=history.history['val_loss']
    v_acc=history.history['val_acc']
    show2(t_loss,t_acc,v_loss,v_acc)


func1()


