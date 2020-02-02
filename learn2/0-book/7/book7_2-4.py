# coding=utf-8

# 使用一维卷积神经网络处理IMDB情感分类问题

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers,models
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard

# import tensorflow as tf
# # 配置gpu训练时内存分配，应该单独学习gpu资源管理，合理分配gpu资源，才能更好的利用，tensorflow还没能在工具层处理这问题，所以才必须在代码中进行配置
# config = tf.ConfigProto(log_device_placement=False)    # 是否打印设备分配日志
# config.gpu_options.per_process_gpu_memory_fraction=0.5 # 设置每个gpu应该拿出多少容量给进程使用
# config.operation_timeout_in_ms=15000   # terminate on long hangs
# sess = tf.InteractiveSession("", config=config)
# init = tf.global_variables_initializer()
# sess.run(init)


max_features = 10000
maxlen = 500  # 序列长度
batch_size = 32 # 序列个数

(train_x, train_y),(test_x, test_y) = imdb.load_data(num_words=max_features)

# print(train_x[0])

# 将每个序列进行反转
# train_x = [x[::-1] for x in train_x]
# test_x = [x[::-1] for x in test_x]

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

def getModel():
    model = models.Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=maxlen, name='embed'))   # 嵌入层,序列向量字典(10000,128)
    model.add(layers.Conv1D(32, 7, activation='relu'))  # 添加一个1D卷积层，卷积窗口长度7，32个特征
    model.add(layers.MaxPooling1D(5))                   # 添加一个1D池化层，池化窗口长度5
    model.add(layers.Conv1D(32, 7, activation='relu')) 
    model.add(layers.GlobalMaxPooling1D())              # 
    # model.add(layers.Dropout(0.5))  # 增加一个dropout层，减小过拟合
    # model.add(layers.Dense(10, activation='relu'))
    # model.add(layers.Dense(1))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'] )
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'] )

    return model

def getCallBacks():
    callbacks = [
        TensorBoard(    # 实例化tensorBoard可视化回调函数
            log_dir = 'my_log_dir_7_2-4',   # 日志文件保存位置
            histogram_freq = 1,             # 没多少轮之后记录激活直方图📊
            embeddings_freq = 1,             # 没多少轮之后记录嵌入数据，即嵌入向量形状
            embeddings_data = train_x[:100].astype("float32")   # 不加上会报：ValueError：To visualize embeddings, embeddings_data must be provided.
        )
    ]
    return callbacks

def run(model):
    history = model.fit(
        train_x,train_y,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=getCallBacks()
    )
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


if __name__ == '__main__':
    model=getModel()
    history=run(model)
    t_loss=history.history['loss']
    t_acc=history.history['acc']
    v_loss=history.history['val_loss']
    v_acc=history.history['val_acc']
    show2(t_loss,t_acc,v_loss,v_acc)


