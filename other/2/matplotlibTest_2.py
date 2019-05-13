#coding=utf-8

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
import pprint



a=np.linspace(0, np.pi, 200)
#ax=plt.subplot(111)

r=1
x0=r*np.cos(a)
y0=r*np.sin(a)

#plt.plot(r*np.cos(x), r*np.sin(x))
#plt.figure()	#定义一个窗口
#plt.plot(x0, y0)
plt.plot(a, y0)
#plt.scatter(x,y3,c=y3)
#plt.scatter(a[:,0],a[:,1],c=a[:,1])
plt.show()	#显示

