#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：adjust_mask_conpare.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2023/2/16 16:26 
'''

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from imageio import imread
from get_metrics.niqe import niqe
from get_metrics.FID import *

from PIL import Image
def Partial_magnification(pic, target, location='lower_right', ratio=2):
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_h]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''
    w, h = pic.shape[1], pic.shape[0],

    target_x, target_y = target[0], target[1]
    target_w, target_h = target[2], target[3]
    cv2.rectangle(pic, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
    new_pic = pic[target_y:target_y + target_h, target_x:target_x + target_w]
    new_pic = cv2.resize(new_pic, (target_w*ratio, target_h*ratio), interpolation=cv2.INTER_CUBIC)
    if location == 'lower_right':
        pic[h-1-target_h*ratio:h-1, w-1-target_w*ratio:w-1] = new_pic
    elif location == 'lower_left':
        pic[h-1-target_h*ratio:h-1, 0:target_w*ratio] = new_pic
    elif location == 'top_right':
        pic[0:target_h*ratio, w-1-target_w*ratio:w-1] = new_pic
    elif location == 'top_left':
        pic[0:target_h*ratio, 0:target_w*ratio] = new_pic
    elif location == 'center':
        pic[int(h/2-target_h*ratio/2):int(h/2+target_h*ratio/2),
            int(w/2-target_w*ratio/2):int(w/2+target_w*ratio/2)] = new_pic
    return pic

def main():
    idx =  [115,
    138,
    194,
    252,
    347,
    421,
    658,
    671,
    694,
    714,
    718,
    740,
    861,
    914,
    927,
    1078,
    1104,
    1322,
    1444,
    1456,
    1650,
    1798,
    1857]
    paths = [r'D:\w\autodl\CeleA_SR_without_ours\lq',r'D:\w\autodl\CeleA_SR_without_ours\hq']
    path1 = [r'D:\w\autodl\CeleA_SR_with_ours\hq',r'D:\w\SR-TEST\HR\Test_CelebA-HQ-gt']
    target = [[120, 193, 250 - 120, 295 - 193], 'lower_right', 2]
    white = np.ones((512, 30, 3), dtype=np.int8) * 255
    for i, name in enumerate(idx):
        name = str(name)+'.jpg'
        pic = cv2.imread(os.path.join(paths[0], name))
        temp = np.zeros_like(pic)
        temp1 = np.zeros_like(pic)
        for j, path in enumerate(paths) :
            pic = cv2.imread(os.path.join(path,name))
            print(os.path.join(path,name))
            pic = Partial_magnification(pic=pic, target=target[0], location=target[1], ratio=target[2])
            if j==0:
                temp = pic
            else:
                temp = np.hstack((temp,white, pic))
        for j,path in enumerate(path1) :
            pic = cv2.imread(os.path.join(path,name))
            pic = Partial_magnification(pic=pic, target=target[0], location=target[1], ratio=target[2])
            if j==0:
                temp1 = pic
            else:
                temp1 = np.hstack((temp1,white, pic))
        compare = np.hstack((temp,white, temp1))
        os.makedirs(f'D:\w\SR-TEST\compare/mask', exist_ok=True)
        cv2.imwrite(f'D:\w\SR-TEST\compare/mask/{name}', compare)

def down_upsampling(picpath,dowmpath,size=16):
    os.makedirs(dowmpath,exist_ok=True)
    pics = os.listdir(picpath)
    for i,name in enumerate(tqdm(pics)):
        path = os.path.join(picpath,name)
        pic = cv2.imread(path)
        temp =cv2.resize(pic,(size,size))
        cv2.imwrite(os.path.join(dowmpath,name),temp)

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = Image.open(image1).convert('L')
    image2 = Image.open(image2).convert('L')
    # image1 = Image.open(image1)
    # image2 = Image.open(image2)
    x_1 = np.array(image1).reshape(1, -1)
    x_2 = np.array(image2).reshape(1, -1)
    con_sim = cosine_similarity(x_1, x_2).item()
    return con_sim

def get_ssim_psnr(im1, im2):
    img1 = imread(im1)
    img2 = imread(im2)
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    ssim = SSIM(img1, img2, multichannel=True)
    psnr = PSNR(img1, img2)
    # print(ssim, psnr)
    # del img1, img2
    return ssim,psnr

if __name__ == '__main__':
    # main()
    HRpic = r'D:\w\SR-TEST\HR\Test_CelebA-HQ-gt'
    LRpic = r'D:\w\SR-TEST\LR\Test_CelebA-HQ_test'
    SRpic = r'D:\w\autodl\CeleA_SR_with_ours\stage'
    for i in [16,32,64,128,256,512]:
        s = 0
        niq=0
        ssim = 0
        psnr = 0
        lr_path = r'D:\w\SR-TEST\down/lr/lr' + str(i)
        if not os.path.exists(lr_path):
            down_upsampling(LRpic, lr_path, i)
        gt_path = r'D:\w\SR-TEST\down/hr/hr' + str(i)
        if not os.path.exists(gt_path):
            down_upsampling(HRpic, gt_path, i)
        sr_path = r'D:\w\SR-TEST\down/sr/sr' + str(i)
        if not os.path.exists(sr_path):
            down_upsampling(SRpic, sr_path, i)
        pics = os.listdir(gt_path)

        print(f'------size={i}=lr vs hr---------')
        for _, pic in enumerate(tqdm(pics)):
            niq += niqe(np.array(Image.open(os.path.join(sr_path, pic)).convert('LA'))[:, :, 0]).item()
        #     s = get_ssim_psnr(os.path.join(lr_path, pic), os.path.join(gt_path, pic))
        #     ssim = ssim + s[0]
        #     psnr = psnr + s[1]
        # print(f'    ssim: ',ssim / len(pics))
        # print(f'    psnr: ',psnr / len(pics))
        print(f'    niqe: ', niq / len(pics))


        print(f'------size={i}=stage vs hr------')
        niq=0
        ssim = 0
        psnr = 0
        for _, pic in enumerate(tqdm(pics)):
            niq += niqe(np.array(Image.open(os.path.join(sr_path, pic)).convert('LA'))[:, :, 0]).item()
        #     s = get_ssim_psnr(os.path.join(sr_path, pic), os.path.join(gt_path, pic))
        #     ssim = ssim + s[0]
        #     psnr = psnr + s[1]
        # print(f'    ssim: ',ssim / len(pics))
        # print(f'    psnr: ',psnr / len(pics))
        print(f'    niqe: ', niq / len(pics))

        # fid_gl = calculate_fid_given_paths(paths=[gt_path, lr_path], batch_size=50, cuda='', dims=2048)
        # fid_gs = calculate_fid_given_paths(paths=[gt_path, sr_path], batch_size=50, cuda='', dims=2048)
        # print(f'gl:gs={i}   {fid_gl}  {fid_gs}')




