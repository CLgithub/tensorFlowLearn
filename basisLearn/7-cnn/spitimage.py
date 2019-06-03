#coding=utf-8
from PIL import Image
import sys
import math
import os
import numpy as np

#分解图片

def spitIm(image,name):
    width,height=image.size
    w=math.trunc(-width/4)
    new_image_length=height
    for i in range(4):
        new_image=Image.new(image.mode, (new_image_length, new_image_length), color='white')
        new_image.paste(image,(w*i,0))
        im=new_image.convert('L')
        #im=np.array(im)
        #print(im.shape)
        #im=Image.fromarray(im)

        x=len(os.listdir('./new_num'))
        im.save('./new_num/'+str(name[i])+'-'+str(x),'PNG')

for imageName in os.listdir('./number_images'):
    if('.' not in imageName):
        im=Image.open('./number_images/'+imageName)
        spitIm(im,imageName)
