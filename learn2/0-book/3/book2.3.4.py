#coding=utf-8

import numpy as np

'''
张量变形    reshape
    元素总个数相同(shape后各个元素相乘，数量一样)，形状不同
特殊的张量变形  转置    transpose
    对形状进行倒序 
'''
print('----------------------2.3.4 张量变形')
arr1=np.array([[-1,0],[1,2],[3,4]])
print(arr1) 
print(arr1.shape) 

arr2=arr1.reshape((6,1))
print(arr2) 
print(arr2.shape) 

arr3=arr2.reshape((2,1,3,1))
print(arr3) 
print(arr3.shape) 

print('---------转置')
arr4=np.transpose(arr3)
print(arr4) 
print(arr4.shape) 
'''
    张量的运算其实就是几何变形，而深度神经网络完全由一系列的张量运算组成，所有深度神经网络可以看作是非常复杂的高维空间的几何变换,最终将复杂的高度折叠的数据流行找到简洁的表示
'''
