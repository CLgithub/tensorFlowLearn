#coding=utf-8

# 猫🐱 狗🐶 图片分类器，使用预训练的卷积神经网络：使用VGG16进行特征提取1：将数据输入到VGG16的卷积基中，得到输出，将改输出输入到模型中

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

original_dataset_dir='/Users/l/develop/clProject/tensorFlowLearn/learn2/0-book/5/data/dogs-vs-cats/train'   #原始数据集解压目录的路径
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

# 定义取出特征函数
def extract_features(directory, sample_count, batch_size=20):
    datagen=ImageDataGenerator(rescale=1./255) 
    conv_base = VGG16(  # 构建卷积基
        weights='imagenet', # 指定模型初始化的权重检查点
        include_top=False,  # 指定模型最后是否包含密集连接分类器
        input_shape=(150,150,3) # 输入到网络中的图像张量的形状（可选），如果不传，网络可以处理任意形状的输入
        )
    # print(conv_base.summary())
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for input_shape, labels_batch in generator:
        features_batch = conv_base.predict(input_shape)
        features[i * batch_size : (i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >=sample_count:
            break
    return features, labels

# 利用卷积基进行数据预处理
def cdate():
    # 分别抽取训练集、校验集、测试集的特征
    train_features, train_labels = extract_features(train_dir, 2000) # train_features.shape=(samples, 4, 4, 512)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    # (samples, 4, 4, 512)要将特征输入到密集连接层，首先展平为(sample,4*4*512=8192)
    train_features = np.reshape(train_features, (2000, 4*4*512))
    validation_features = np.reshape(validation_features, (1000, 4*4*512))
    test_features = np.reshape(test_features, (1000, 4*4*512))
    return train_features,train_labels, validation_features,validation_labels, test_features,test_labels

# 构建模型
def getModel():
    #搭建模型，只需要密集连接层
    model=models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4*4*512 ))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc']
            )
    return model

def run(model,train_features,train_labels,validation_features,validation_labels):
    # 训练
    history = model.fit(
        train_features,
        train_labels,
        epochs=15,
        batch_size=20,
        validation_data=(validation_features, validation_labels)
        )
    # 保存训练结果
    model.save('cats_and_dogs_small_5.3.1.h5')  #保存模型
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
    #copyData()
    train_features,train_labels, validation_features,validation_labels, test_features,test_labels=cdate()
    model=getModel()
    history=run(model,train_features,train_labels,validation_features,validation_labels)
    t_loss=history.history['loss']
    t_acc=history.history['acc']
    v_loss=history.history['val_loss']
    v_acc=history.history['val_acc']
    show2(t_loss,t_acc,v_loss,v_acc)


func1()

