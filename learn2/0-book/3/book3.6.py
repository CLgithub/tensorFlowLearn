#coding=utf-8
import numpy as np
from tensorflow import keras
from keras.datasets import boston_housing
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

#导入数据
(train_data, train_targets),(test_data, test_targets)=boston_housing.load_data()

#print(train_data[0])

#数据处理
#数据标准化
mean=train_data.mean(axis=0)    #获取特征平均值
train_data -=mean               #输入数据的每个特征，减去特征平均值
std=train_data.std(axis=0)
train_data /=std                #再除以标准差
test_data -= mean
test_data /=std     #注意，用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。在工作流程中，你不能使用在测试数据上计算得到的任何结果，即使是像数据标准化这么简单的事情也不行



#构建网络模型
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #编译模型
    model.compile(
        loss='mse',     #损失函数，均方误差(MSE,mean squared error)
        optimizer='rmsprop',
        metrics=['mae']     #衡量标准,平均绝对误差(MAE mean absolute error) ,思考：此处若使用acc会怎样
    ) 
    return model

#训练模型，训练的同时监测模型在验证数据集上的表现，避免过拟合
#由于数据较少，所以使用k折交叉验证方法划分训练集和验证集
# k折验证
k=4
num_val_samples=len(train_data) // k
num_epochs=80   #经过绘图，发现80次时损失最小，之后会过拟合
all_mae_histories=[]
for i in range(k):
    print('processing fold #', i)
    index_a=i*num_val_samples
    index_b=(i+1)*num_val_samples
    val_data=train_data[index_a: index_b]   #截取第i个分区的数据
    val_targets=train_targets[index_a:index_b]

    partial_train_data=np.concatenate(  #concatenate 串联
        [ train_data[:index_a], train_data[index_b:] ],
        axis=0
    )
    partial_train_targets=np.concatenate(
        [ train_targets[:index_a], train_targets[index_b:] ],
        axis=0
    )
    model=build_model()
    history=model.fit(
        partial_train_data, 
        partial_train_targets, 
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=16,
        verbose=0
    )
    mae_history=history.history['val_mean_absolute_error']
    #print(mae_history)
    #val_mse, val_mae=model.evaluate(val_data, val_targets, verbose=0)   #在验证数据集上评估模型
    #all_scores.append(val_mae)
    all_mae_histories.append(mae_history)

test_mse_score,test_mae_score=model.evaluate(test_data, test_targets)   #在验证数据集上评估模型
print(test_mae_score)
#print(all_scores)
#print(np.mean(all_scores))

#绘制图形
#print(all_mae_histories)

average_mae_history=[ np.mean( [x[i] for x in all_mae_histories] ) for i in range(num_epochs) ] #求得num_epochs次各次平均
#print(average_mae_history)
#将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线
def smooth_curve(points, factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history=smooth_curve(average_mae_history[10:])

plt.plot( range(1, len(smooth_mae_history)+1), smooth_mae_history )
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

    
