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
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=1, help='Number of iteration times')
args = parser.parse_args()

model = MyChain()
serializers.load_npz('../model/'+args.model, model)

data = []

# パーツを検出させたい画像の読み込み
img = ImageOps.invert(Image.open('../test_data/t10.jpg').convert('L'))

"""
鼻を中心として、両目間の距離の1.5倍を一辺の長さとする正方形で切り抜き、
全体の2/3を100*100に拡大する正規化を施している

ランドマークの情報がない状況で
この作業を行うにはどうしたらよいか？

TODO1:仮の正規化を考えて実装してみる
"""
width,height = img.size
size = min(width,height)

img = img.crop((width//2-size//2,height//2-size//2,width//2+size//2,height//2+size//2))

img = np.array(img,dtype=np.float32) / 256.0

# ランドマークの座標情報を持たない画像に前処理をする
def preprocess(data):

    return data

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        img_data.append(input_data)
    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))
    return x

dst = preprocess(img)
x = mini_batch_data_without_t(dst)
y = model(x)

show_img_and_landmark(x.data[0][0],y.data[0].reshape((landmark,2)))
