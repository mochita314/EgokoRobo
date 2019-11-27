import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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
            #print(w2, h2, width + w2, height + h2)
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
            print(w2, h2, width + w2, height + h2)
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

    lr_lst.append(parts[29].x - parts[2].x)
    lr_lst.append(parts[14].x - parts[29].x)

    ud_lst.append(parts[30].y - parts[27].y)
    ud_lst.append(parts[8].y - parts[30].y)

    if lr_lst[0] < lr_lst[1]:
        min_num_lr = 0
    else:
        min_num_lr = 1
    
    if ud_lst[0] < ud_lst[1]:
        min_num_ud = 0
    else:
        min_num_ud = 1
    
    total_width_lr = sum(lr_lst) 
    total_width_ud = sum(ud_lst) 

    for i in range(len(lr_lst)):
        lr_lst[i] = lr_lst[i]/total_width_lr*2

    for i in range(len(ud_lst)):
        ud_lst[i] = ud_lst[i]/total_width_ud*2

    return lr_lst,min_num_lr,ud_lst,min_num_ud,total_width_lr,total_width_ud

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

if __name__ == '__main__':

    print("正面を向いてください")
    lr_lst, min_num_lr,ud_lst,min_num_ud,total_width_lr,total_width_ud = detect_face_angle()
    print("次の写真を撮ります")
    sleep(3)
    print("作りたい方向を向いてください")
    lr_lst2, min_num_lr2,ud_lst2,min_num_ud2,total_width_lr2,total_width_ud2 = detect_face_angle()

    angle = 2

    print(ud_lst[0],ud_lst2[0],ud_lst[1],ud_lst2[1])

    if ud_lst2[0]*total_width_ud2/(ud_lst[0]*total_width_ud)< 0.9:
        print("上を向いています")
        angle = 0
        key_ratio = ud_lst2[0]*total_width_ud2/(ud_lst[0]*total_width_ud)
    elif ud_lst2[1]*total_width_ud2/(ud_lst[1]*total_width_ud)< 0.9:
        print("下を向いています")
        angle = 1
        key_ratio = ud_lst2[1]*total_width_ud2/(ud_lst[1]*total_width_ud)

    im = Image.open('../img/ilust.png')

    # 左右方向の向きのみを合わせる
    im = ImgSplit_ver(im)
    for i in range(len(lr_lst)):
        width,height = im[i].size
        if i==0:
            im[i] = im[i].resize((int(width*lr_lst[min_num_lr2]),height))
    dst = get_concat_h(im[0],im[1])
    if min_num_lr2==1 and lr_lst[1]<0.85:
        dst = ImageOps.mirror(dst)

    dst.save("../img2/img" + datetime.now().strftime("%Y%m%d_%H%M%S%f_") +".jpg", "JPEG")

    #必要な場合、鼻を逆向きにしてのせる
    #この後ここに髪の毛を追加する必要がある
    
    #上下方向の向きを調整する必要がある場合 
    if angle != 2:

        im = Image.open('../img/face_frame.png')
        
        im = ImgSplit_hor(im)
        for i in range(2):
            width,height = im[i].size
            if i==angle and angle==0:
                im[i] = im[i].resize((width,int(max(0.6,key_ratio*0.8)*height)))
            elif i==angle:
                im[i] = im[i].resize((width,int(max(0.6,key_ratio)*height)))

        dst = get_concat_v(im[0],im[1])

        dst = ImgSplit_ver(dst)
        for i in range(len(lr_lst)):
            width,height = dst[i].size
            if i==min_num_lr2:
                dst[i] = dst[i].resize((int(width*lr_lst2[i]),height))
        dst = get_concat_h(dst[0],dst[1])

        #各パーツ配置

        




        
        dst.save("../img2/img" + datetime.now().strftime("%Y%m%d_%H%M%S%f_") +".jpg", "JPEG")