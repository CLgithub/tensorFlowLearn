# coding=utf-8

# 使用keras中的循环神经网络，对IMDB电影评论进行处理  LSTM

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

# 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
config.operation_timeout_in_ms=15000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

max_features = 10000
maxlen = 500  # 序列长度
batch_size = 32 # 序列个数

(train_x, train_y),(test_x, test_y) = imdb.load_data(num_words=max_features)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

def getModel():
    model = models.Sequential()
    model.add(layers.Embedding(max_features, 32))	# 嵌入层,序列向量字典(10000,32)
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'] )

    return model


def run(model):
    history = model.fit(
        train_x,train_y,
        epochs=10,
        batch_size=32,
        validation_split=0.2)
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
    model=getModel()
    history=run(model)
    t_loss=history.history['loss']
    t_acc=history.history['acc']
    v_loss=history.history['val_loss']
    v_acc=history.history['val_acc']
    show2(t_loss,t_acc,v_loss,v_acc)

func1()


