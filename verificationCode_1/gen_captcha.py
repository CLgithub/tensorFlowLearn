# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import cv2
import os
 
# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','    v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','    V','W','X','Y','Z']
 
# 验证码长度为4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
 
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text(char_set=number)
    captcha_text = ''.join(captcha_text)    #用空字符串来连接4个字符集合
    captcha = image.generate(captcha_text)  #生成图片对象
    #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
    captcha_image = Image.open(captcha)     #获取图片信息
    captcha_image = np.array(captcha_image) #将图片转行成np数组
    return captcha_text, captcha_image
 
if __name__ == '__main__':
    #保存路径
    path = './trainImage'
    # path = './validImage'
    for i in range(3000):
        text, image = gen_captcha_text_and_image()
        fullPath = os.path.join(path, text + ".jpg")
        cv2.imwrite(fullPath, image)
        print "{0}/10000".format(i)
    print "/nDone!"


