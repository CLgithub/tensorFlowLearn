#coding=utf-8

'''
该学习内容源自tensorflow官网教程，在对人工智能、机器学习、深度学习、神经网络、tensorflow、keras的关系有了进一步的了解后，再次学习该教程

https://www.tensorflow.org/tutorials/keras/basic_classification
'''
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Hei']  #指定显示的中文字体
plt.rcParams['axes.unicode_minus']=False  #用来正常显示中文标签
#有中文出现的情况，需要u'内容'

#导入数据集
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = [u'T恤衫', u'裤子', u'套头毛衣', u'连衣裙', u'外套', u'凉鞋', u'衬衫', u'运动鞋', u'包包', u'踝靴']
#预处理，不知道为何这么做
train_images=train_images/255.0
test_images=test_images/255.0

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.figure(figsize=(10,10)) #开一个小窗口
for i in range(25):
    plt.subplot(5,5,i+1)    #甚至横竖行
    #plt.xticks([])          #设置不显示像素横
    #plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i]) #cmap=plt.cm.binary 灰度图显示
    plt.xlabel(class_names[train_labels[i]]) #设置在横方向上显示标题
plt.show()
'''

def createModel():
    #设置层，构建模型
    model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),          #L1,先将数据拉平,28*28=784
        keras.layers.Dense(128, activation=tf.nn.relu),      #L2,将拉平后的数据传给L2层128个神经元，使用relu进行激励
        keras.layers.Dense(10, activation=tf.nn.softmax)    #L3结果层,将L2层的结果传给10个神经元，激励函数为softmax，得到具有10个概率得分的数组
    ])
    #编译模型
    model.compile(
        optimizer=tf.train.AdamOptimizer(), #设置优化器
        loss='sparse_categorical_crossentropy', #设置损伤函数
        metrics=['accuracy']    #设置监控指标=精准度
    )
    return model

#训练模型
def learn1(model):
    model.fit(train_images, train_labels, epochs=1)
    return model
#训练模型并自动保存
def learnsave1(model,checkpoint_path):
    #checkpoint_path="training_1/cp.ckpt"
    cp_callback=tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            #verbose=5
            period=5       #设置没5个周期保存一次检查点
    )
    model.fit(train_images, train_labels, epochs=15,
            validation_data=(test_images, test_labels),
            callbacks=[cp_callback]
    )
    return model

#加载检查点
def load1(model,checkpoint_path):
    checkpoint_dir=os.path.dirname(checkpoint_path)
    latest=tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

def plot_image(i, predictions_array, lables, images):
    predictions_array, lable, img=predictions_array[i], lables[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary) #cmap=plt.cm.binary显示图片，以灰度图显示
    predicted_label=np.argmax(predictions_array)    #获取到各个可能性
    if predicted_label==lable:
        color='red'
    else:
        color='blue'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
        100*np.max(predictions_array), 
        class_names[lable]), 
        color=color)

def plot_value_array(i, predictions_array, layers):
    predictions_array, lable=predictions_array[i], layers[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot=plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label=np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')
    thisplot[lable].set_color('red')

def getPre():
    model=createModel() #创建模型
    checkpoint_path1='training_1/cp.ckpt'
    checkpoint_path2='training_2/cp-{epoch:04d}.ckpt'
    checkpoint_path3='./training_3/my_checkpoint'
    checkpoint_path4='./training_4/my_model.h5'

    '''保存检查点'''
    #model=learnsave1(model, checkpoint_path2) #训练模型并保存模型到检查点
    #model_new=load1(model,checkpoint_path2)    #加载检查点
    '''手动保存保存检查点'''
    #model_new=learn1(model) #训练模型
    #model_new.save_weights(checkpoint_path3)
    #model_new.load_weights(checkpoint_path3)
    '''保存整个模型'''
    #model_new=learn1(model) #训练模型
    #model_new.save(checkpoint_path4)
    model_new=keras.models.load_model(checkpoint_path4)
    model_new.compile( optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    predictions=model_new.predict(test_images)  #做出预测
    test_loss,test_acc=model_new.evaluate(test_images, test_labels) #评估模型
    return predictions, test_loss, test_acc

def test1(i,predictions):
    plt.figure(figsize=(6,3))   #开一个小窗
    plt.subplot(1,2,1)  
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

def test2(predictions):
    num_rows=5  #5行
    num_cols=3
    num_images=num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))  #开一个小窗
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

predictions,test_loss,test_acc=getPre()
print(test_acc)
#test1(0,predictions)
#test2(predictions)

