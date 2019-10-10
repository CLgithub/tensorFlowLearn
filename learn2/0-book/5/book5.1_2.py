#coding=utf-8
# 检验卷积工作原理 page=99

import numpy as np

# 被取出的其中一个图片块
image_back=np.array( [ 
            [[2],[1],[1]],
            [[1],[2],[3]],
            [[1],[2],[3]],
            ] )
print(image_back.shape) # (3,3,2)
# 某个通道的卷积核
con_kernel=np.array(
            [[2]]
            )
print(con_kernel.shape)
# 这个图块与该特征的输出，所有图块与该特征的输出即该特征的响应图
out=np.dot(image_back,con_kernel)
print(out.shape)


