#condig=utf-8

import numpy as np
from tensorflow import keras
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Hei']  #指定显示的中文字体
plt.rcParams['axes.unicode_minus']=False  #用来正常显示中文标签
#有中文出现的情况，需要u'内容'

#导入数据
np_load_old=np.load
np.load=lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
np.load=np_load_old

word_index=imdb.get_word_index()    #单词所对应的索引
reverse_word_index=dict(   #索引对应单词
    [(value,key) for (key, value) in word_index.items()]
)
'''
decode_review=' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
    #索引减去3,因为0，1，2是“padding”、“start of sequence”、“unknown”
)
'''

'''
由于影评数据长度不同，需要转换为能输入神经网络的张量，有两种方法
    1.填充，使其具有相同的长度
    2.对列表进行one-hot编码,将参差不齐长度的list转换为长度相同的0，1数组
        (此方法思路新奇，值得思考)
        得到的不是原list，而是原list在那些索引处出现了
'''

# 定义数据处理方法
def vectorize_sequences(sequences, dimension=10000):
    results=np.zeros((len(sequences), dimension))
    for i, j in enumerate(sequences):
        results[i, j] = 1.
    return results
'''
arr1=np.array([(3,5),(6,3,2,1),(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)])
#enumerate 
aa=vectorize_sequences(arr1)
print(aa)
'''

#数据处理
x_train=vectorize_sequences(train_data)
y_train=np.asarray(train_labels).astype('float32')

x_test=vectorize_sequences(test_data)
y_test=np.asarray(test_labels).astype('float32')

x_val=x_train[:10000]   # 预留10000个做验证集
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]

#构建网络
def getModel():
    model=models.Sequential()   #创建模型
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))    #第一层，输入(s,10000),输出(s,16)
    model.add(layers.Dense(16, activation='relu' ))
    model.add(layers.Dense(1, activation='sigmoid' ))
    #编译模型   指定损失计算方法(损失函数),权重更新方法(优化器),衡量方法
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    history=model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val,y_val)
    )
    return history
def getModel2():
    model=models.Sequential()   #创建模型
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))    #第一层，输入(s,10000),输出(s,16)
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu' ))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid' ))
    #编译模型   指定损失计算方法(损失函数),权重更新方法(优化器),衡量方法
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    history=model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val,y_val)
    )
    return history

def getlossAndAcc(history):
    history_dict=history.history
    loss_values=history_dict['val_loss']
    acc_values=history_dict['val_acc']
    return loss_values, acc_values

history1=getModel()
loss1,acc1=getlossAndAcc(history1)
history2=getModel2()
loss2,acc2=getlossAndAcc(history2)

epochs=range(1, len(loss1)+1)
def show1(loss1, loss2, acc1, acc2):
    plt.plot(epochs, loss1, 'b', label='Original model')
    plt.plot(epochs, loss2, 'c', label='Droput-regularizerd model')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.clf()   #清空图像
    plt.plot(epochs, acc1, 'b', label='Original model')
    plt.plot(epochs, acc2, 'c', label='Droput-regularizerd model')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

show1(loss1, loss2, acc1, acc2)

#用测试集来评测模型的准确率
#results=model.evaluate(x_test,y_test)
#print(results)
#用训练好的模型来评判测试集
arr1=model.predict(x_test)
print(arr1)
#print(test_data[1])
p=[]
for i in test_data[0]:
    p.append(reverse_word_index[i])
#print(p)

'''
对于二分类的经验及小结，请参阅3.4.7 （59 page）
'''
