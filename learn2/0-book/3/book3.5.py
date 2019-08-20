#coding=utf-8
import numpy as np
from tensorflow import keras
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

#导入数据
np_load_old=np.load
np.load=lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(train_data, train_labels),(test_data, test_labels)=reuters.load_data(num_words=10000)
np.load=np_load_old

# 定义数据处理方法
def vectorize_sequences(sequences, dimension=10000):
    results=np.zeros((len(sequences), dimension))
    for i, j in enumerate(sequences):
        results[i, j] = 1.
    return results
'''
将标签向量化有两种方法：1.补全，2.one-hot编码
keras内置了这种方法
'''
#数据向量化
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)
#标签向量化
one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

#构建神经网络
model=models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))  #可以尝试增加或减少维度，增加或减少隐层
model.add(layers.Dense(46, activation='softmax'))
#编译模型   指定损失计算方法(损失函数),权重更新方法(优化器),衡量方法
model.compile(
    loss='categorical_crossentropy',    #损失函数   绝对_叉熵
    optimizer='rmsprop',    #优化器, SGD的一个变种
    metrics=['accuracy']    #衡量方法
    )

#预留数据
x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

#训练
history=model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
    )

#用测试集评估模型loss和acc
results=model.evaluate(x_test, one_hot_test_labels)
print(results)

#用训练好的模型去查看测试集的结果
predictions=model.predict(x_test)
print(predictions[0])
print(np.argmax(predictions[0]))


#图像展示
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

acc=history.history['acc']
val_acc=history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 通过这个例子应该学到
