#coding=utf-8

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
import pprint

x = np.linspace(-5, 5, 1000)	#定义x的范围-1,1，并且取50个点
#y = x*x	#y与x的关系

'''
a=np.random.random((5,2))
print(a)
print(a[:,0])
'''

y1=1/(1+pow(math.e,-x))  #sigmoid函数，激励函数之一 a=1/(1+e^-z)
y2=(1-pow(math.e,-x))/(1+pow(math.e,-x))  #双s函数
y3=np.cos(pow(x,1))
#y3=pow(x,3)


#plt.figure()	#定义一个窗口
plt.plot(x, y3)	#画出x与y的关系
#plt.scatter(x,y1,c=x)
#plt.scatter(x,y2,c=x)
#plt.scatter(x,y3,c=y3)
#plt.scatter(a[:,0],a[:,1],c=a[:,1])
plt.show()	#显示

