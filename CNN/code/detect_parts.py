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
parser.add_argument('testfile', type=str, help='testfile created by imglab tool')
parser.add_argument('model', type=str, help='model file')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=1, help='Number of iteration times')
args = parser.parse_args()

model = MyChain()
serializers.load_npz('../model/'+args.model, model)

data = []

# パーツを検出させたい画像の読み込み
img = ImageOps.invert(Image.open('../../input_img/ilust.png').convert('L'))
img = np.array(img,dtype=np.float32) / 256.0

data.append({'img':img})

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        img_data.append(input_data['img'])
    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))
    return x

x = mini_batch_data_without_t(img)
y = model(x)

for i in range(args.batchsize):
    show_img_and_landmark(x.data[i][0],y.data[i].reshape((landmark,2)))






