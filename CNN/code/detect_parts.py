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

"""
TODO1:入力は最終的には手書き画像になるので、それのコントラスト、シャープネスなどの
　　　 調整を自動化させる
TODO2:検出したパーツを切り抜いて、それぞれ画像として保存するコードを完成させる

"""

# パーツを検出させたい画像の読み込み
img = ImageOps.invert(Image.open('../test_data/t11.jpg').convert('L'))

# 鼻の位置を検出する
# ここさえ綺麗にできればうまくいく
# あとで高速化を試みること
def detect_nose(img,shade=args.shade):
    width = img.size[0]
    height = img.size[1]

    chin_candidate = list(range(int(width*0.45),int(width*0.55)))
    
    # まず顎を検出する
    for y in reversed(list(range(0,height))):
        for x in chin_candidate:
            arr = np.array([])
            r,g,b = img.convert('RGB').getpixel((x,y))
            arr = np.append(arr,[r,g,b])
            if np.sum(arr) > 300: #黒が検出された場合
                chin = [x,y]
                break
        else:
            continue
        break

    print(chin)

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
                    if num==1+shade:
                        mouth = y
                        print(mouth)
                    break
            else:
                if x == chin[0]+9: #そのy座標にひとつも黒い点がなかった場合
                    flag = 0
                continue
            break
    
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
def preprocess(img):

    width,height = img.size

    n2 = detect_nose(img)

    center = [n2[0],n2[1]-8]
    length = int(width * 0.35)

    img = np.array(img.crop(get_square(center,length)),dtype=np.float32) / 256.0

    return img

def zoom_and_flip(img):

    # imgはnumpyの配列なので、widthとheightの取得の仕方は先程と異なる
    width = img.shape[1]
    height = img.shape[0]

    center = (width/2,height/2)

    angle = 0
     
    scale0 = 100.0 / (width * 2.0 / 3.0)
    scale = scale0 * random.uniform(0.9,1.1)

    matrix = cv2.getRotationMatrix2D(center,angle,scale) + np.array([[0, 0, -center[0] + 50 + random.uniform(-3, 3)], [0, 0, -center[1] + 50 + random.uniform(-3, 3)]])
    
    img = cv2.warpAffine(img,matrix,(100,100))

    if random.randint(0,1)==1:
        img = cv2.flip(img,1)
    
    return img

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        data = zoom_and_flip(input_data)
        img_data.append(data)
    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))
    return x

dst = preprocess(img)
x = mini_batch_data_without_t(dst)
y = model(x)

show_img_and_landmark(x.data[0][0],y.data[0].reshape((landmark,2)))

# -*- パーツの画像をそれぞれ切り出して保存する機能をこの後に実装する -*- 
