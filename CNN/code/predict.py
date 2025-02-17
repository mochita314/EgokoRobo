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

"""
https://github.com/TadaoYamaoka/gochiusa
"""

parser = argparse.ArgumentParser(description='Predict facial landmark')
parser.add_argument('testfile', type=str, help='testfile created by imglab tool')
parser.add_argument('model', type=str, help='model file')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=1, help='Number of iteration times')
args = parser.parse_args()

model = MyChain()
serializers.load_npz('../model/'+args.model, model)

test_data = []
load_image(args.testfile, test_data)

def mini_batch_data(train_data):
    img_data = []
    t_data = []
    for j in range(args.batchsize):
        data = data_augmentation(train_data[random.randrange(len(train_data))])
        img_data.append(data['img'])
        t_data.append(data['parts'])
        # for debug
        #show_img_and_landmark(data['img'], data['parts'])

    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))

    t = Variable(np.array(t_data, dtype=np.float32))
    t = F.reshape(t, (args.batchsize, landmark*2))

    return x, t

for i in range(args.iteration):
    x, t = mini_batch_data(test_data)
    y = model(x)
    loss = F.mean_squared_error(y, t)
    print("test loss = {}".format(loss.data))

    for i in range(args.batchsize):
        show_img_and_landmark(x.data[i][0], y.data[i].reshape((landmark,2)))