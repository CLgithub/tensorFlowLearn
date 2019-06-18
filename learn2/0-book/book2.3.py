#coding=utf-8

'''
2.3 神经网络的齿轮⚙️a :张量
'''
import numpy as np

print('------------------------------ 2.3  ')
'''------------------------------ 2.3
深度神经网络学到的所有变换可以简化为数值数据张量的一些张量运算（tensor operation）

keras.layers.Dense(512, activation='relu')   #通过Dense来构建网络,这个层可以理解为一个函数，输入一个2D张量，返回另一个2D张量，即输入张量的新表示，具体而言这个函数如下：
    output=relu(dot(W, input)+b)
    W是一个2D张量，b是一个向量(1D张量)
    dot是点积运算
    + 形状相同,各位子上逐个元素相加
    relu(x)=max(x,0)    即x小于等于0则为0，大于0这为x
'''
print('------------------------------ 2.3.1 逐元素运算  ')
'''------------------------------ 2.3.1 逐元素运算
relu和加法运算都是逐元素运算，即该运算独立应用于张量的每个元素
'''
#relu运算简单实现
def naive_relu(x):
    assert len(x.shape) == 2    #x是2D张量  assert断言,若正确则继续运行，若错误这抛出AssertionError错误
    x=x.copy()
    for i in range(x.shape[0]):     #沿着x的第一根轴方向遍历
        for j in range(x.shape[1]):
            x[i,j]=max(x[i,j],0)
    return x
arr1=np.array([[1,2],[3,-1],[4,2]])    #shape=(3,2)
print(naive_relu(arr1))
#print(naive_relu(arr1))
print('----')

#加法运算简单实现
def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape 
    x=x.copy()
    for i in range(x.shape[0]):     #沿着x的第一根轴方向遍历
        for j in range(x.shape[1]):
            x[i,j]+=y[i,j]
    return x

arr2=np.array([[1,2],[3,-1],[4,2]])    #shape=(3,2)
print(naive_add(arr1,arr2))
print(arr1+arr2)

print('------------------------------ 2.3.2 广播  ')
'''------------------------------ 2.3.1 广播
当两个形状不同的张量发送逐元素运算时，较小的张量会被"广播",以匹配较大张量的形状，广播分为两步：
    (1) 向较小的张量添加轴（叫做广播轴）,是其维度ndim与较大的张量相同
    (2) 将较小张量沿着新轴重复，使其形状与较大张量相同
    此两步不会发生在内存中，只限于思维模式
    
'''
arr1=np.array([[1,2],[3,-1],[4,2]])    #shape=(3,2)
print(arr1)
print(arr1.shape)
print('--------------------')
arr2=np.array([3,2])
print(arr2)
print(arr2.shape)
'''
arr1+arr2,给arr2加一个轴，变成[[3,2]],然后沿着加的这个轴方向重复,变成[[3,2],[3,2],[3,2]]
'''
print(arr1+arr2)    #可以简单实现这个函数
'''
通常的：如果一个张量的形状是(a,b, ..., n,n+1, ..., m),另一个张量的形状是(n, n+1, ..., m)，那么可以利用广播对它们做两个张量的逐元素运算,广播会自动应用与从a到n-1的轴
'''
x=np.random.random((64,3,32,10))
y=np.random.random((32,10))
z=np.maximum(x, y)
print(z.shape)  #z的形状与x相同，均为(64,3,32,10)
