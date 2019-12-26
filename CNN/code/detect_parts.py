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

from model import *

parser = argparse.ArgumentParser(description='Predict facial landmark')
#parser.add_argument('testfile', type=str, help='testfile created by imglab tool')
parser.add_argument('model', type=str, help='model file')
parser.add_argument('shade',type=int,help='the number of shade')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
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

def zoom_and_flip(img):

    # imgはnumpyの配列なので、widthとheightの取得の仕方は先程と異なる
    width = img.shape[1]
    height = img.shape[0]

    center = (width/2,height/2)

    angle = 0    
    scale0 = 100.0 / (width * 2.0 / 3.0)
    scale = scale0 * random.uniform(0.9,1.1)

    scale1 = scale/scale0 # あとで座標変換に必要なので取り出しておく

    matrix = cv2.getRotationMatrix2D(center,angle,scale) + np.array([[0, 0, -center[0] + 50 + random.uniform(-3, 3)], [0, 0, -center[1] + 50 + random.uniform(-3, 3)]])
    img = cv2.warpAffine(img,matrix,(100,100))

    # 反転されたかどうかの記録(=flipが0か1か)は座標変換に必要
    flip = 0
    if random.randint(0,1)==1:
        img = cv2.flip(img,1)
        flip = 1
    
    return img,scale1,flip

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        data,scale1,flip = zoom_and_flip(input_data)
        img_data.append(data)
    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))
    return x,scale1,flip

# 検出された座標を、最初の入力である元画像上での座標に置き換える
def coordinate_transformation(parts):
    parts2 = [[0 for i in range(2)] for j in range(12)]
    for i in range(len(parts)):
        parts2[i][0] = int(2*0.7/3*width*parts[i][0] + (0.5-scale1/3)*0.7*width+n2[0]-width*0.35)
        parts2[i][1] = int(2*0.7/3*width*parts[i][1] + (0.5-scale1/3)*0.5*width+n2[1]-width*0.35)
        if flip == 1:
            parts_converted = [[0 for i in range(2)] for j in range(12)]
            for i in range(len(parts2)):
                parts_converted[i][0] = width - parts2[i][0]
            parts_lst = [parts2[4],parts2[5],parts2[6],parts2[7],parts2[0],parts2[1],parts2[2],parts2[3],parts2[8],parts2[9],parts2[11],parts2[10]]
        else:
            parts_lst = parts2
    return parts_lst

# パーツを検出させたい画像の読み込み
img = ImageOps.invert(Image.open('../test_data/t11.jpg').convert('L'))

# 仮の正規化をしてパーツを検出する
width,height = img.size
dst,n2 = preprocess(img,width,height)
x,scale1,flip = mini_batch_data_without_t(dst)
y = model(x)

parts = y.data[0].reshape((landmark,2))
parts_lst = coordinate_transformation(parts)

show_img_and_landmark(x.data[0][0],y.data[0].reshape((landmark,2)))
