# coding=utf-8

# 在IMDB数据上使用Embedding层和分类器，利用预训练的词嵌入

import os
import numpy as np
from keras.preprocessing.text import Tokenizer      # 导入keras中的分词器
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, Dense, Embedding
from keras import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

imdb_dir = './data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')

glove_dir = './data/glove.6B'

# 进行分词
sWordLen = 100 # 每条评论限定长度
training_samples = 200      # 仅在200个样本上训练
validation_samples = 10000  # 在10000个样本上验证
allWordLen = 10000           # 只考虑数据集中前10000个常见的单词
embedding_dim = 100 # 使用的是glove.6B.100d.txt

# 处理原始数据，将原始数据转换成词嵌入关联方式的数据，
# 并且获取词嵌入层的权重array，
# 该层输入(samples, sWordLen), 权重矩阵index_nparray(allWordLen, embedding_dim), , 输出(samples, sWordLen, embedding_dim)

def readData(r_dir):
    labels=[]
    texts=[]
    for label_type in ['neg','pos']:
        dir_name = os.path.join(r_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return labels, texts

def getData():
    # 1.读取数据 
    labels, texts=readData(train_dir)


    # 2.获取到分词字典 word_index
    tokenizer = Tokenizer(num_words=allWordLen)
    tokenizer.fit_on_texts(texts)
    word_index=tokenizer.word_index     # 获取到分词字典表 {'the': 1, 'is': 2, 'a': 3,....'canada': 295}


    # 3.采用 词嵌入 关联 方式，关联出数据 index
    # tokenizer.texts_to_matrix(texts, mode='binary')   # one-hot 关联 方式
    sequences = tokenizer.texts_to_sequences(texts)     # 词嵌入 关联 方式，将字符串转换为整数索引组成的列表
    # print(sequences)    # [[66, 4, 3, 67,...],[235, 38, 24, ..],...[., 65, 16]]
    data = pad_sequences(sequences, maxlen=sWordLen)  # 每条评论限定长度 转换为np 数组数据(索引)
    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])  # 每个数据的索引
    np.random.shuffle(indices)          # 随机打乱
    data = data[indices]
    labels = labels[indices]

    train_x = data[ :training_samples]
    train_y = labels[ :training_samples]
    val_x = data[training_samples: training_samples+validation_samples]
    val_y = labels[training_samples: training_samples+validation_samples]


    # 4.读取glove.6B.100d.txt，获得单词到密集向量的关系 word_nparray
    word_nparray = {}    # 定义 {[单词1]:密集张量1, [单词2]:密集张量2, ... }集合
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_nparray[word] = coefs
    f.close()


    # 5.通过 word_index & word_nparray 获取到能加载到Embedding层的 index_nparray
    index_nparray = np.zeros((allWordLen, embedding_dim)) #定义能加载到Embedding层中的嵌入矩阵,形状为(allWordLen, embedding_dim)的矩阵
    for word,i in word_index.items():
        if i < allWordLen:   # 必须是在max_words以内的分词字典索引，按理说不会超出，因为word_index分词字典获取时已经做了限制
            embedding_vector = word_nparray.get(word)    # 获取到该单词对应的密集张量
            if embedding_vector is not None:    # 如果该密集张量不为None
                index_nparray[i] = embedding_vector  # 将单词索引i对应上了密集张量
    return train_x,train_y, val_x,val_y, index_nparray

def getModel(index_nparray):
    model = models.Sequential()
    model.add(Embedding(allWordLen, embedding_dim, input_length=sWordLen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    # 将预训练的词嵌入加载到 Embedding 层中，
    # 注意：此处与卷积神经网络不同，卷积神经网络是直接将预训练层直接加入到模型中，而此处只是设置权重矩阵
    model.layers[0].set_weights([index_nparray]) 
    model.layers[0].trainable = False   # 设置Embedding层不可训练

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def run(model, train_x, train_y, val_x, val_y):
    history = model.fit(
        train_x,train_y,
        epochs=10,
        batch_size=32,
        validation_data=(val_x, val_y)
        )

    model.save_weights('model_book6_2-2.h5')
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

def assModel():
    labels, texts=readData(test_dir)

    tokenizer = Tokenizer(num_words=allWordLen)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    test_x = pad_sequences(sequences, maxlen=sWordLen)
    test_y = np.asarray(labels)

    index_nparray = np.zeros((allWordLen, embedding_dim))
    model = getModel(index_nparray)
    model.load_weights('model_book6_2-2.h5')
    print(model.evaluate(test_x, test_y))

def func1():
    train_x,train_y, val_x,val_y, index_nparray=getData()

    model=getModel(index_nparray)

    print(model.layers[0].input.shape)  # Embedding 层输入形状
    print(model.layers[0].weights)      # Embedding 层权重
    print(model.layers[0].output.shape) # Embedding 层输出形状

    history=run(model, train_x, train_y, val_x, val_y)

    t_loss=history.history['loss']
    t_acc=history.history['acc']
    v_loss=history.history['val_loss']
    v_acc=history.history['val_acc']
    # show2(t_loss,t_acc,v_loss,v_acc)

func1()

assModel()







