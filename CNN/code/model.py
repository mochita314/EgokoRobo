#!/usr/bin/python

"""
https://github.com/TadaoYamaoka/gochiusa
のコードをもとに書いています
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


imgsize = 100
landmark = 12

def out_size(in_size, ksize, poolsize):
    return math.ceil((in_size - ksize + 1)  / poolsize)

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

def get_center_and_length(parts):
    center = [(parts["n1"][0]+parts["n2"][0])/2,(parts["n1"][1]+parts["n2"][1])/2]
    length = math.sqrt((parts["le2"][0]-parts["re2"][0])**2 + (parts["le2"][1]-parts["re2"][1])**2)*0.75
    return center,length

def get_square(center,length):
    # 鼻が中心にきて、かつ一辺の長さが両目端間の距離の1.5倍の正方形の
    # 左上の座標(x_1,y_1)と右下の座標(x_2,y_2)を取り出す
    return (center[0]-length,center[1]-length,center[0]+length,center[1]+length)
    # x_1, y_1, x_2, y_2

def load_image(xmlfile,data):
    # データを読み込んで、パーツの位置の座標を格納する

    try:
        tree = ET.parse('../train_data/'+xmlfile)
    except:
        tree = ET.parse('../test_data/'+xmlfile)

    root = tree.getroot()
    images = root.find("images")

    for image in list(images):
        # グレースケールに変換
        try:
            img = ImageOps.invert(Image.open('../train_data/'+image.get("file")).convert('L'))
        except:
            img = ImageOps.invert(Image.open('../test_data/'+image.get("file")).convert('L'))

        for box in list(image):
            # パーツの座標を格納する辞書
            parts = {}
            for part in list(box):
                parts[part.get("name")] = [int(part.get("x")), int(part.get("y")), 1]
            
            notfound = False
            for label in ("re1", "re2", "re3", "re4", "le1", "le2", "le3", "le4", "n1", "n2", "m1", "m2"):
                if label not in parts:
                    print("not exist {} in box:{}, file:{}".format(label, box,image.get("file")))
                    notfound = True
            if notfound:
                continue
            
            #正規化のために切り出す正方形の中心の座標と一辺の長さの半分を取得
            center,length = get_center_and_length(parts)
            #取得した中心座標と辺の長さで画像を切り取る
            cropped_img = np.array(img.crop(get_square(center,length)),dtype=np.float32) / 256.0
            
            #切り取った正方形に合わせてランドマークの座標変換
            for part in parts.values():
                part[0] -= center[0]-length
                part[1] -= center[1]-length
            
            data.append({'img': cropped_img, 'parts' : parts})

def data_augmentation(data):
    img = data['img']
    parts = data['parts']

    width = img.shape[1]
    height = img.shape[0]

    center = (width / 2, height / 2)

    dx = parts["re1"][0] - parts["le1"][0]
    dy = - parts["re1"][1] + parts["le1"][1]
    angle0 = math.degrees(math.atan(dy / dx))

    # 2/3の範囲を100*100にする
    scale0 = 100.0 / (width * 2.0/ 3.0)

    # ランダムに変形を加える
    angle = random.uniform(-45,45)-angle0
    scale = scale0 * random.uniform(0.9, 1.1)

    # 変形後の原点
    #x0 = width / 3.0 * scale

    # アフィン変換
    #  回転、拡大の後に、回転の中心が(50, 50)になるように平行移動
    #  平行移動にランダムな値を加える
    matrix = cv2.getRotationMatrix2D(center, angle, scale) + np.array([[0, 0, -center[0] + 50 + random.uniform(-3, 3)], [0, 0, -center[1] + 50 + random.uniform(-3, 3)]])
    dst = cv2.warpAffine(img, matrix, (100, 100))

    # ランドマーク座標をnumpyの配列に変換
    parts_np = np.array([
        parts["re1"], parts["re2"], parts["re3"],parts["re4"],
        parts["le1"], parts["le2"], parts["le3"], parts["le4"],
        parts["n1"], parts["n2"], 
        parts["m1"], parts["m2"]
        ], dtype=np.float32)

    # ランドマークの座標変換
    parts_converted = parts_np.dot(matrix.T) / 100.0

    # ランダムに反転
    if random.randint(0, 1) == 1:
        #print("yes")
        dst = cv2.flip(dst, 1)
        for i in range(len(parts_converted)):
            parts_converted[i][0] = 1.0 - parts_converted[i][0]

        parts_converted = np.array([
            parts_converted[4], parts_converted[5], parts_converted[6], parts_converted[7],# C
            parts_converted[0], parts_converted[1], parts_converted[2], parts_converted[3], # R -> L
            parts_converted[8], parts_converted[9],# L -> R
            parts_converted[11], parts_converted[10],# M
            ], dtype=np.float32)

    # 変換されたデータを返す
    return {'img' : dst, 'parts' : parts_converted}

def show_img_and_landmark(img, parts):
    plt.imshow(1.0 - img, cmap='gray')
    #print(parts)

    #右目は赤
    for t in parts[0:4]:
        plt.plot(t[0]*100, t[1]*100, 'or')
    #左目は緑
    for t in parts[4:8]:
        plt.plot(t[0]*100, t[1]*100, 'og')
    #鼻は青
    for t in parts[8:10]:
        plt.plot(t[0]*100, t[1]*100, 'ob')
    #口は黄色
    for t in parts[10:12]:
        plt.plot(t[0]*100, t[1]*100, 'oy')
    plt.axis([0, 100, 100, 0])
    plt.show()