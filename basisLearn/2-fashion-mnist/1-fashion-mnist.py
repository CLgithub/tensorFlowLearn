#coding=utf-8

# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
import keras
# Helper libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

#导入数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)
# 各种服饰标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#class_names = ['T恤衫', '裤子', '套头毛衣', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包包', '踝靴']

#图像预处理，除以255，转换成0～1
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
plt.figure()
print(train_images[1][0])
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10,10)) #开一个小窗口
for i in range(25):
    plt.subplot(5,5,i+1)    #甚至横竖行
    plt.xticks([])          #设置不显示像素横
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary) #cmap=plt.cm.binary 灰度图显示
    plt.xlabel(class_names[train_labels[i]]) #设置在横方向上显示标题
plt.show()
'''

#设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),         #第一层，将图像格式从二维数组(28*28)转换成一维数组(28*28=784)
    keras.layers.Dense(128, activation=tf.nn.relu),     #第一个密集层，128个神经元，采用relu方法
    keras.layers.Dense(10, activation=tf.nn.softmax)    #第二个密集层，10个神经元，采用softmax（判断各个概率）
])

#编译模型
#损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
#优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
#指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

#训练模型
#model.fit(train_images, train_labels, epochs=5) #开始训练
#model.save_weights( 'my_checkpoint')    #手动保存权重
#model.save('my_model.h5')                #保存整个模型
#new_model = keras.models.load_model('my_model.h5') #加载保存的模型,tensorflow与karas版本有问题
#new_model.summary()

#保存训练出的权重
#1 检查点回调法
checkpoint_path = "training_1/cp.ckpt"  
checkpoint_dir = os.path.dirname(checkpoint_path)
# 创建一个检查点
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
#训练模型，顺便保存到检查点
def learn1(cp_callback,epochs):
    model.fit(train_images, train_labels,  epochs = epochs,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

checkpoint_path_2 = "training_2/cp-{epoch:04d}.ckpt"  
#checkpoint_path_2 = "training_2/cp-0015.ckpt"  
checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)
cp_callback_2 = keras.callbacks.ModelCheckpoint(checkpoint_path_2, save_weights_only=True, verbose=1, period=3)
def learn2(cp_callback,epochs):
    model.fit(train_images, train_labels,  epochs = epochs,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary) #展示图片，按灰度图
    predicted_label = np.argmax(predictions_array)  #取出最大的
    if predicted_label == true_label:   #如果标签相等，
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
        100*np.max(predictions_array), 
        class_names[true_label]), 
        color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def test1(i):
    predictions = model.predict(test_images)    #对测试集进行预测试
    print(predictions[i])
    print(np.argmax(predictions[i]))
    print('预测：'+class_names[np.argmax(predictions[i])])
    print('正确：'+class_names[ test_labels[i] ])
    plt.figure()
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def test2(i,model1):
    predictions = model1.predict(test_images)    #对测试集进行预测试
    plt.figure(figsize=(6,3))   #开一个小窗
    plt.subplot(1,2,1)  #放在一行两列的第一个位置
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)  #放在一行两列的第二个位置
    plot_value_array(i, predictions, test_labels )
    plt.show()

def test3():
    predictions = model.predict(test_images)    #对测试集进行预测试
    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

learn1(cp_callback,5)   #学习并将学习到的权重保存到checkpoint_path中
#model.load_weights('training_2/cp-0015.ckpt') #model加载检查点的权重
#test2(5,model) #测试训练成果
test1(0)
