#!/usr/bin/python

import numpy as np
import chainer
from chainer import Function, Variable
from chainer import serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import ImageOps
import math
import sys
import argparse

import random
import cv2
from matplotlib import pylab as plt

from time import sleep

from model import *

parser = argparse.ArgumentParser(description='Predict facial landmark')
#parser.add_argument('testfile', type=str, help='testfile created by imglab tool')
parser.add_argument('model', type=str, help='model file')
parser.add_argument('image', type=str, help='image file')
parser.add_argument('shade',type=int,help='the number of shade')
parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=1, help='Number of iteration times')
args = parser.parse_args()

model = MyChain()
serializers.load_npz('../model/'+args.model, model)

# 鼻の位置を検出する
# ここさえ綺麗にできればうまくいく
# あとで高速化を試みること
def detect_nose(img,width,height,shade=args.shade):

    chin_candidate = list(range(int(width*0.45),int(width*0.55)))
    
    # まず顎を検出する
    for y in reversed(list(range(0,height))):
        for x in chin_candidate:
            arr = np.array([])
            r,g,b = img.convert('RGB').getpixel((x,y))
            arr = np.append(arr,[r,g,b])
            if np.sum(arr) > 300: #黒が検出された場合
                chin = [x,y]
                print(chin)
                break
        else:
            continue
        break

    #次に口を検出する
    flag = 1
    num=0
    for y in reversed(list(range(0,chin[1]))):
        for x in list(range(chin[0]-10,chin[0]+10)):
            arr = np.array([])
            r,g,b = img.convert('RGB').getpixel((x,y))
            arr = np.append(arr,[r,g,b])
            if np.sum(arr) > 500: #黒検出
                if flag == 0:
                    flag = 1
                    num+=1
                    # 口元の影があるかどうかによって条件判定が変わる
                    if num==1+shade:
                        mouth = y
                        print(mouth)
                    break
            else:
                if x == chin[0]+9: #そのy座標にひとつも黒い点がなかった場合
                    flag = 0
                continue
            break

    # 最後に鼻を検出する
    for y in reversed(list(range(0,mouth-8))):
        for x in list(range(chin[0]-10,chin[0]+10)):
            arr = np.array([])
            r,g,b = img.convert('RGB').getpixel((x,y))
            arr = np.append(arr,[r,g,b])
            if np.sum(arr) > 500: #黒検出
                n2 = [x,y]
                print(n2)
                break
        else:
            continue
        break

    return n2

# 検出した鼻の位置をもとに、画像に仮の正規化を施す
def preprocess(img,width,height,shade=args.shade):

    n2 = detect_nose(img,width,height,shade)

    center = [n2[0],n2[1]-8]
    length = int(width * 0.35)

    img = np.array(img.crop(get_square(center,length)),dtype=np.float32) / 256.0

    return img,n2

def zoom(img):

    # imgはnumpyの配列なので、widthとheightの取得の仕方は先程と異なる
    width = img.shape[1]
    height = img.shape[0]

    center = (width/2,height/2)

    angle = 0    
    scale0 = 100.0 / (width * 2.0 / 3.0)
    scale = scale0 * random.uniform(0.9,1.1)

    scale1 = scale0/scale # あとで座標変換に必要なので取り出しておく

    matrix = cv2.getRotationMatrix2D(center,angle,scale) + np.array([[0, 0, -center[0] + 50 + random.uniform(-3, 3)], [0, 0, -center[1] + 50 + random.uniform(-3, 3)]])
    img = cv2.warpAffine(img,matrix,(100,100))
    
    return img,scale1

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        data,scale1 = zoom(input_data)
        img_data.append(data)
    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))
    return x,scale1

# 検出された座標を、最初の入力である元画像上での座標に置き換える
def coordinate_transformation(parts):

    parts2 = [[0 for i in range(2)] for j in range(12)]

    for i in range(len(parts)):
        parts2[i][0] = int(2*0.7/3.0*width*parts[i][0] + (0.5-scale1/3.0)*0.7*width+n2[0]-width*0.35)
        parts2[i][1] = int(2*0.7/3.0*width*parts[i][1] + (0.5-scale1/3.0)*0.7*width+n2[1]-width*0.35)

    return parts2

# 座標変換後の座標をもとに、各パーツの画像を切り出して保存する
def crop_parts(img,parts):

    re_width = parts[0][0] - parts[1][0]
    re_height = parts[3][1] - parts[2][1]
    re_center = [(parts[0][0]+parts[1][0])//2,(parts[2][1]+parts[3][1])//2]

    le_width = parts[5][0] - parts[4][0]
    le_height = parts[7][1] - parts[6][1]
    le_center = [(parts[4][0]+parts[5][0])//2,(parts[6][1]+parts[7][1])//2]

    nose_height = parts[9][1] - parts[8][1]
    nose_width = abs(parts[8][0] - parts[9][0])
    nose_center = [(parts[8][0]+parts[9][0])//2,(parts[8][1]+parts[9][1])//2]

    mouth_height = abs(parts[10][1] - parts[11][1])
    mouth_width = parts[11][0] - parts[10][0]
    mouth_center = [(parts[10][0]+parts[11][0])//2,(parts[10][1]+parts[11][1])//2]

    re1 = [re_center[0]-(re_width//2+5),re_center[1]-(re_height//2+5)]
    re2 = [re_center[0]+(re_width//2+5),re_center[1]+(re_height//2+5)]
    print('re1:{},re2:{}'.format(re1,re2))

    le1 = [le_center[0]-(le_width//2+5),le_center[1]-(le_height//2+5)]
    le2 = [le_center[0]+(le_width//2+5),le_center[1]+(le_height//2+5)]

    nose1 = [nose_center[0]-(nose_width//2+5),nose_center[1]-(nose_height//2+5)]
    nose2 = [nose_center[0]+(nose_width//2+5),nose_center[1]+(nose_height//2+5)]

    mouth1 = [mouth_center[0]-(mouth_width//2+5),mouth_center[1]-(mouth_height//2+5)]
    mouth2 = [mouth_center[0]+(mouth_width//2+5),mouth_center[1]+(mouth_height//2+5)]

    right_eye = img.crop((re1[0],re1[1],re2[0],re2[1]))
    left_eye = img.crop((le1[0],le1[1],le2[0],le2[1]))
    nose = img.crop((nose1[0],nose1[1],nose2[0],nose2[1]))
    mouth = img.crop((mouth1[0],mouth1[1],mouth2[0],mouth2[1]))

    right_eye.save("../../original_img/right_eye.jpg","JPEG")
    left_eye.save("../../original_img/left_eye.jpg","JPEG")
    nose.save("../../original_img/nose.jpg","JPEG")
    mouth.save("../../original_img/mouth.jpg","JPEG")


# パーツを検出させたい画像の読み込み
img = ImageOps.invert(Image.open('../test_data/'+args.image).convert('L'))

# 仮の正規化をしてパーツを検出する
width,height = img.size
dst,n2 = preprocess(img,width,height)
x,scale1 = mini_batch_data_without_t(dst)
y = model(x)

parts = y.data[0].reshape((landmark,2))

# 元の画像での座標のリスト
parts_lst = coordinate_transformation(parts)
print(parts_lst)

# パーツごとに切り抜いた画像の保存

crop_parts(img,parts_lst)




# 検出されたランドマークの表示
show_img_and_landmark(x.data[0][0],y.data[0].reshape((landmark,2)))
