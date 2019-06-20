#coding=utf-8

import numpy as np

print('------------------------------ 2.3.3 张量点积  ')
'''------------------------------ 2.3.3 张量点积
张量点积,
    若是2D张量，其实就是矩阵乘法，z=dot(x,y)，z.shape=(x.shape[0],y.shape[1]),要求x.shape[1]=y.shape[0]
    不过张量把这种计算推广到了更多维,切不限于2D*2D，可以是2D*1D，

    推广到更多维度，
        张量x与张量y，两个张量要做点积
            1.当其中一个是标量(0D)时，形状等于另一个的形状
            2.当两个都是2D张量时，x和y元素个数必须相同，也就是形状一样,得到一个标量(0D)
            3.当y为1D，切x为非标量，则形状为x.shape[:-1]
        多维的情况, 要实现x能点积y，x的最后一个轴要和y的倒数第2个轴相同，即x.shape[-1]==y.shape[-2]，得到的点积的形状为(x去掉最后一位,y去掉倒数第2位) 即：(x.shape[:-1],y.shape[:-2],y.shape[-1:]) 若位数不够，则不取

    张量点积不同于*，*是逐元素运算，形状不同需要广播

'''
print('-----------------2D')
def d2d():
    print('----arr1')
    arr1=np.array([[3,2,1],[1,2,1]])
    print(arr1)
    print(arr1.shape)
    print('----arr2')
    arr2=np.array([[1],[2],[1]])
    print(arr2)
    print(arr2.shape)
    print('----dot(arr1,arr2)')
    arr3=np.dot(arr1,arr2)
    print(arr3)
    print(arr3.shape)
    print('----arr2_')
    arr2_=np.array([1,2,1])
    print(arr2_)
    print(arr2_.shape)
    print('----dot(arr1,arr2_)')
    arr4=np.dot(arr1,arr2_)
    print(arr4)
    print(arr4.shape)
d2d()
print('-----------------推广到所有可能')
arr1=np.array(
    [
        [
            [
                [1,2,2,1],
                [1,2,2,1],
            ],
            [
                [1,2,1,1],
                [1,2,2,1],
            ],
            [
                [1,2,2,1],
                [1,2,2,1],
            ]
        ],
        [
            [
                [1,2,2,1],
                [1,2,2,1],
            ],
            [
                [1,2,2,1],
                [2,2,1,1]
            ],
            [
                [1,2,2,1],
                [2,2,2,1]
            ]
        ]
    ]
)
arr1=np.array([4,1,3,2])
#arr1=np.array(4)
print(arr1)
print(arr1.shape)
arr2=np.array(
    [
        [
            [
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ]
            ],
            [
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ]
            ],
            [
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ],
                [ 1,2 ]
            ]
        ]
    ]
)

arr2=np.array([2,1,3,4])
#arr2=np.array(2)
print(arr2)
print(arr2.shape)

arr3=np.dot(arr1,arr2)
#print(arr3)
print('arr3_shape',arr3.shape)

#print(arr1.shape[:-1],arr2.shape[:-2],arr2.shape[-1] )
a=-1
b=-2
c=-1
if arr1.ndim==0:    #形状为arr2.shape
    a=0
    b=c
if arr2.ndim==0:    #形状为arr1.shape
    a=None
if arr1.ndim>0 and arr2.ndim==1:    #形状为arr1.shape[:-1]
    c=1
if arr1.ndim==1 and arr2.ndim==1:   #形状为()
    a=0
    b=c-1
print(arr1.shape[:a],arr2.shape[:b],arr2.shape[c:] )


print('--*')
print(arr1*arr2)
