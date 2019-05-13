# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import cv2
import os

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 

char_set=number+ALPHABET

# 图像大小
IMAGE_HEIGHT = 60  #80
IMAGE_WIDTH = 160  #250
MAX_CAPTCHA = 4
#trainImagePath='/Users/l/develop/clProject/tensorFlowLearn/verificationCode_1/trainImage/vcode/1-700'
trainImagePath='/Users/l/develop/clProject/tensorFlowLearn/verificationCode_1/trainImage/GzEyeNetIndexIm'
 
# 验证码长度为4个字符
def random_captcha_text(char_set=char_set, captcha_size=MAX_CAPTCHA):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
 
# 生成字符对应的验证码
def gen_captcha_text_and_image_r():
    image = ImageCaptcha()
    captcha_text = random_captcha_text(char_set=char_set)
    captcha_text = ''.join(captcha_text)    #用空字符串来连接4个字符集合
    captcha = image.generate(captcha_text)  #生成图片对象
    #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
    captcha_image = Image.open(captcha)     #获取图片信息
    captcha_image = np.array(captcha_image) #将图片转行成np数组
    return captcha_text, captcha_image

def func1():
    for i in range(10):
        text, image = gen_captcha_text_and_image_r()
        fullPath = os.path.join(trainImagePath, text + ".jpg")
        print(fullPath)
        #cv2.imwrite(fullPath, image, [int(cv2.IMWRITE_JPEG_QUALITY),9])
        image=cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT) )
        cv2.imwrite(fullPath, image )
        #print "{0}/10000".format(i)
    print("/nDone!")
 
if __name__ == '__main__':
	func1()
	'''
    image=ImageCaptcha()
    captcha = image.generate('1234')
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    print captcha_image
	'''
