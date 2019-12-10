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

