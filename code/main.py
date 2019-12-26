#!/usr/bin/python

"""
detect_parts.pyでパーツを検出して切り出した画像を保存したのち、
その画像をもとにして、角度に合わせて回転をさせる
"""

import cv2
import dlib
import os
import numpy as np

from PIL import Image, ImageOps
import sys

from datetime import datetime
from time import sleep

# 画像を縦半分で2つに分割する
def ImgSplit_ver(im):
    # 読み込んだ画像を真ん中で縦長二つに分割する
    width,height = im.size
    width /= 2

    buff = []
    # 縦の分割枚数
    for h1 in range(1):
        # 横の分割枚数
        for w1 in range(2):
            w2 = w1 * width
            h2 = h1 * height
            c = im.crop((w2, h2, width + w2, height + h2))
            buff.append(c)
    return buff

def ImgSplit_hor(im):
    # 読み込んだ画像を真ん中で横長二つに分割する
    width,height = im.size
    height /= 2

    buff = []
    # 縦の分割枚数
    for h1 in range(2):
        # 横の分割枚数
        for w1 in range(1):
            w2 = w1 * width
            h2 = h1 * height
            c = im.crop((w2, h2, width + w2, height + h2))
            buff.append(c)
    return buff

def detect_face_angle():

    predictor_path = "../shape_predictor_68_face_landmarks.dat"

    # download trained model
    if not os.path.isfile(predictor_path):
        os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        os.system("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    dets = detector(frame,1)

    lr_lst = []
    ud_lst = []

    for d in dets:
        parts=predictor(frame,d).parts()
    
    try:
        lr_lst.append(parts[29].x - parts[2].x)
        lr_lst.append(parts[14].x - parts[29].x)

        ud_lst.append(parts[30].y - parts[27].y)
        ud_lst.append(parts[8].y - parts[30].y)
    except:
        print("no face")
        exit()

    if lr_lst[0] < lr_lst[1]:
        min_num_lr = 0 #右向き
    else:
        min_num_lr = 1 #左向き
    
    if ud_lst[0] < ud_lst[1]:
        min_num_ud = 0 #上向き
    else:
        min_num_ud = 1 #下向き
    
    total_width_lr = sum(lr_lst) 
    total_width_ud = sum(ud_lst) 

    for i in range(len(lr_lst)):
        lr_lst[i] = lr_lst[i]/total_width_lr*2

    for i in range(len(ud_lst)):
        ud_lst[i] = ud_lst[i]/total_width_ud*2

    return lr_lst,min_num_lr,ud_lst,min_num_ud,total_width_lr,total_width_ud

#2枚の画像を横に連結
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

#2枚の画像を縦に連結
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def np_pic_paste(part,part_x,part_y,pic):
    arr1 = np.array([])
    arr2 = np.array([])
    for x in range(part_x,part_x+part.size[0]+1):
        for y in range(part_y,part_y+part.size[1]+1):
            r,g,b = pic.convert('RGB').getpixel((x,y))
            arr1 = np.append(arr1,[x,y,r,g,b])
    pic.paste(part,(part_x,part_y))
    arr1 = arr1.astype('int64')
    num=0
    for x in range(part_x,part_x+part.size[0]+1):
        for y in range(part_y,part_y+part.size[1]+1):
            r2,g2,b2 = pic.convert('RGB').getpixel((x,y))
            arr2 = np.append(arr2,[x,y,r2,g2,b2])
            if np.sum(arr1[5*num:5*num+5]) < np.sum(arr2[5*num:5*num+5]):
                try:
                    pic.putpixel((arr1[5*num],arr1[5*num+1]),(arr1[5*num+2],arr1[5*num+3],arr1[5*num+4]))
                except:
                    print(0)
            num+=1
    return pic

def faster_pic_paste(part,part_x,part_y,pic):
    pic.paste(part,(part_x,part_y))
    return pic

if __name__ == '__main__':

    print("正面を向いてください")
    lr_lst, min_num_lr,ud_lst,min_num_ud,total_width_lr,total_width_ud = detect_face_angle()
    print("次の写真を撮ります")
    sleep(2)
    print("作りたい方向を向いてください")
    lr_lst2, min_num_lr2,ud_lst2,min_num_ud2,total_width_lr2,total_width_ud2 = detect_face_angle()

    angle = 2

    if ud_lst2[0]*total_width_ud2/(ud_lst[0]*total_width_ud)< 0.9:
        print("上を向いています")
        angle = 0
        key_ratio = ud_lst2[0]*total_width_ud2/(ud_lst[0]*total_width_ud)
    elif ud_lst2[1]*total_width_ud2/(ud_lst[1]*total_width_ud)< 0.9:
        print("下を向いています")
        angle = 1
        key_ratio = ud_lst2[1]*total_width_ud2/(ud_lst[1]*total_width_ud)

    #ベースとなる輪郭
    im = Image.open('../input_img/face.png')

    #配置するパーツ
    nose = Image.open('../input_img/nose.png')
    right_eye = Image.open('../input_img/right_eye.png')
    left_eye = Image.open('../input_img/left_eye.png')
    mouth = Image.open('../input_img/mouth.png')
    try:
        shadow = Image.open('../input_img/shadow.png')
    except:
        print("no shadow")
        shadow = 1
    
    if angle != 2: #上下方向の向きを調整する必要がある場合
        
        im = ImgSplit_hor(im)
        for i in range(2):
            width,height = im[i].size
            if i==angle and angle==0: #上向きの場合
                im[i] = im[i].resize((width,int(max(0.6,key_ratio*0.8)*height)))

                ratio_nose = 0.15
                ratio = 1

                right_eye = right_eye.resize((right_eye.size[0],int(max(0.6,key_ratio*0.8)*right_eye.size[1])))
                left_eye = left_eye.resize((left_eye.size[0],int(max(0.6,key_ratio*0.8)*left_eye.size[1])))

            elif i==angle: #下向きの場合
                im[i] = im[i].resize((width,int(max(0.6,key_ratio)*height)))

                ratio_nose = 0.5
                ratio = max(0.6,key_ratio)

                right_eye = right_eye.resize((right_eye.size[0],int(max(0.6,key_ratio*0.8)*right_eye.size[1])))
                left_eye = left_eye.resize((left_eye.size[0],int(max(0.6,key_ratio*0.8)*left_eye.size[1])))

        nose_y = int(im[0].size[1] + int(ratio_nose*im[1].size[1])-nose.size[1]/2)
        eye_y = im[0].size[1]+10
        
        dst = get_concat_v(im[0],im[1])
    
    else: #左右の調整だけで良い場合
        dst = im
        nose_y = int(0.75*dst.size[1]-nose.size[1]/2)
        eye_y = int(0.5*dst.size[1])
        ratio = 1
    
    dis_nose_mouth = 10
    mouth_y = int(nose_y + nose.size[1] + dis_nose_mouth*ratio)
    shadow_y = int(nose_y + 110*ratio)

    #左右方向の向きを合わせる
    dst = ImgSplit_ver(dst)
    mouth = ImgSplit_ver(mouth)
    for i in range(len(lr_lst)):
        width,height = dst[i].size
        width2,height2 = mouth[i].size
        
        if i==0: #右向きか左向きかに関わらず、とりあえず右側を圧縮する
            dst[i] = dst[i].resize((int(width*lr_lst2[min_num_lr2]),height))
            mouth[i] = mouth[i].resize((int(width2*lr_lst2[min_num_lr2]),height2))

            right_eye = right_eye.resize((int(lr_lst2[min_num_lr2]*right_eye.size[0]),right_eye.size[1]))

            nose_x = int(width*lr_lst2[min_num_lr2]-nose.size[0]/2)
            right_eye_x = int(nose_x-100*lr_lst2[min_num_lr2]-right_eye.size[0]/2)
            mouth_x = int(nose_x + nose.size[0]/2 - mouth[i].size[0])
        else:
            dst[i] = dst[i].resize((int(width*lr_lst2[abs(1-min_num_lr2)]),height))
            mouth[i] = mouth[i].resize((int(width2*lr_lst2[abs(1-min_num_lr2)]),height2))

    dst = get_concat_h(dst[0],dst[1])
    mouth = get_concat_h(mouth[0],mouth[1])

    left_eye_x = int(nose_x + 60)
    shadow_x = int(nose_x + nose.size[0]/2 - shadow.size[0]/2)

    #パーツ同士の重なりに注意しながら貼り付け
    dst = faster_pic_paste(nose,nose_x,nose_y,dst)
    print("nose complete")
    dst = np_pic_paste(left_eye,left_eye_x,eye_y,dst)
    print("left eye complete")
    dst = np_pic_paste(right_eye,right_eye_x,eye_y,dst)
    print("right eye complete")
    dst = faster_pic_paste(mouth,mouth_x,mouth_y,dst)

    print("mouth complete")
    if shadow != 1:
        dst = faster_pic_paste(shadow,shadow_x,shadow_y,dst)
        print("shadow complete")

    if min_num_lr2==1 and lr_lst[1]<0.85: #顔が左向きの場合
        dst = ImageOps.mirror(dst)
   
    dst.save("../output_img/img" + datetime.now().strftime("%Y%m%d_%H%M%S%f_") +".jpg", "JPEG")