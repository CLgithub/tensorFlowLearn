# coding=utf-8

import numpy as np

a1 = np.array([1,3,5,7,9], 'float64')
a2 = np.array([3,4,5,6,9], 'float64')
a3 = np.array([-3,400,5,6000,9], 'float64')

mean1 = a1.mean(axis=0)
mean2 = a2.mean(axis=0)
mean3 = a3.mean(axis=0)


'''
( (1-5)(1-5)+(3-5)(3-5)+(7-5)(7-5)+(9-5)(9-5) )/5 
=(16+4+4+16)/5
=8
然后开方

( (3-5)(3-5)+(4-5)(4-5)+(6-5)(6-5)+(7-5)(7-5) )/5 
=(4+1+1+4)/5
=2
然后开方
'''

std1 = a1.std(axis=0)	# 均方差
std2 = a2.std(axis=0)	# 均方差
std3 = a3.std(axis=0)	# 均方差

# 先减去平均值，得到每个元素与平均值的差距，而这些差距刚好反应出了均方差，再除以均方差，所以使得数据标准化
a1 -= mean1
a1 /= std1

a2 -= mean2
a2 /= std2

a3 -= mean3
a3 /= std3

print(a1)
print(a2)
print(a3)
