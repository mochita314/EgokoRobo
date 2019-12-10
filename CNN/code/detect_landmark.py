#!/usr/bin/python

"""
学習済みのモデルを利用して
入力画像からパーツを検出するプログラム
"""

import numpy as np
import chainer
from chainer import Function, Variable
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import ImageOps
import math
import sys

import random
import cv2
from matplotlib import pylab as plt

class MyChain(Chain):

    def __init__(self):
        super(MyChain,self).__init__(
            l1=L.Convolution2D(in_channels = 1, out_channels = 16, ksize = 4),
            l2=L.Convolution2D(in_channels = 16, out_channels = 32, ksize = 5),
            l3=L.Convolution2D(in_channels = 32, out_channels = 64, ksize = 5),
            l4=L.Linear((out_size(out_size(imgsize, 4, 2), 5, 2) - 5 + 1)**2*64, 400),
            l5=L.Linear(400, landmark*2)
        )
    
    def __call__(self,x):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        h3 = F.relu(self.l3(h2))
        h3_reshape = F.reshape(h3, (len(h3.data), int(h3.data.size / len(h3.data))))
        h4 = F.relu(self.l4(h3_reshape))
        return self.l5(h4)

