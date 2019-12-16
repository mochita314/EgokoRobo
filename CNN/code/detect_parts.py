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

def data_augumentation_without_t(data):
    img = data['img']
    return {'img':img}

def mini_batch_data_without_t(input_data):
    img_data = []
    for j in range(args.batchsize):
        data = data_augumentation_without_t(data)



