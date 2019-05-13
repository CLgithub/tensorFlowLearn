#coding=utf-8
import numpy as np
from sklearn.datasets import make_moons
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV

#np.random.seed(0)       #设置相同的seed时，每次生成的随机数相同，但是是一次性的
#X, y = make_moons(100, noise=0.1)       #制作样本数据，产生一个随机样本sklearn.datasets.make_moons(n_samples=100    , shuffle=True, noise=None, random_state=None)，n_samples整数型，总的产生的样本点数,shuffle 是否对样本重新洗牌， noise 加到数据里的高斯噪声标准差
c=np.array([0,0,1,1,1,0])   #生成一个一维数组
x=np.linspace(0,5,6)       #在0～5之间取6个数
y=x*x                    
X=np.array([x,y]).T       #得到一个第一行为x，第二行为y的数组，然后转置
plt.scatter(X[:,0], X[:,1], s=40, c=c, cmap=plt.cm.Spectral)  #画出来：横坐标为x,纵坐标为y，颜色按c分布
#plt.show()

# 咱们先顶一个一个函数来画决策边界
def plot_decision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max,h))
    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=c, cmap=plt.cm.Spectral)

#咱们先来瞄一眼逻辑斯特回归对于它的分类效果
clf = LogisticRegressionCV()    #逻辑回归，广义线性分析模型，相当于用一条直线去最接近的分割
clf.fit(X, c)

# 画一下决策边界
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
#plt.show()
####################################


# 剃度下降参数
epsilon = 0.01  #学习率
reg_lambda=0.01 #正则化参数

#向前运算
def c(X,model):
    #model 字典类型，存储两个y=f(x) f()的有两个参数w和b
    W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'],model['b2']
    z1 = X.dot(W1)+b1   #y=f1(x)
    #print('z1:',z1)
    a1 = np.tanh(z1)    #双曲函数
    #print('a1:',a1)
    z2 = a1.dot(W2)+b2  #y=f2(f1(x))
    #print('z2:',z2) #经过函数得到的结果
    exp_scores = np.exp(z2) #以自然常数e为底数的指数函数
    #print('e:',exp_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #axis=1,各行自己相加
    return probs
    

#定义损失函数
# 损失函数：在0和第j类的得分-第yi类的得分的差值+c的结果中找最大的，即找出第j类与yi类的差别的大小(反映的是第j类与第yi类的差别)，再将这些差别大小加起来，从而反映所有类别与实际结果第yi类(yi=0)的吻合度
def calculate_loss(model):
    probs = c(X,model)
    #model 字典类型，存储两个y=f(x) f()的有两个参数w和b
    W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'],model['b2']
    #计算损失
    corect_logprobs=-np.log(probs[range(num_examples), y])  #probs[]取出元素,然后log，公式参考pptpdf第二课21页,判断其结果是否更接近y,更可能是0还是1的损失
    #print('c:',corect_logprobs)
    data_loss=np.sum(corect_logprobs)
    #print('dl:',data_loss)
    #加一下正则化项
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) #square平方
    #print('dl:',data_loss)
    return 1./num_examples * data_loss

num_examples = len(X) # 样本数
nn_input_dim = 2 # 输入维度
nn_output_dim = 2 # 输出的类别个数

def build_model(nn_hdim, num_passes=5000, print_loss=False):
    '''
    1) nn_hdim:隐层节点个数
    2) num_passes:剃度下降迭代次数
    3) print_loss:设定为True的话，每1000次迭代输出一次loss的当前值
    '''
    # 随机初始化一下权重w,b
    #np.random.randn(x1,x2)随机产生x1行x2列的数组，np.sqrt开方
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim,nn_hdim) / np.sqrt(nn_input_dim)   
    b1 = np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim,nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    print('W1:',W1,'b1:',b1,'W2:',W2,'b2:',b2)
    '''
    # 学到的模型
    model = {}
    model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    i=0
    while i<num_passes:
        i+=1
        #for i in xrange(0, num_passes):
        # 向前运算loss
        z1=X.dot(W1)+b1
        #print('z1:',z1)
        a1=np.tanh(z1)
        z2=a1.dot(W2)+b2
        exp_scores=np.exp(z2)
        #probs=exp_scores / np.sum(exp_scores, axis=1, keepdims=True)    #最终的结果
        probs = c(X,model)
        print(probs)
        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        #print(delta3)
    '''
    
#build_model(1,num_passes=1)
np.random.seed(0)
w=np.random.randn(2,1) / np.sqrt(2)
print w
