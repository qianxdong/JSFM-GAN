#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：evaluate.py
@Function:
@Date    ：2022/3/23 20:45 
'''


import utils_image as util
import os
from tqdm import tqdm
# import cv2


def evulate():
    hr_paths = util.get_image_paths(HR_path)
    numbers = len(hr_paths)
    sum_psnr = 0
    max_psnr = 0
    min_psnr = 100
    sum_ssim = 0
    max_ssim = 0
    min_ssim = 1
    for hr_path in hr_paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_name = os.path.basename(hr_path)
        sr_path = os.path.join(SR_path, img_name)
        print(img_name)
        # print(hr_path)
        # print(sr_path)
        img_Hr = util.imread_uint(hr_path, n_channels=n_channels)  # HR image, int8
        img_Sr = util.imread_uint(sr_path, n_channels=n_channels)  # HR image, int8
        psnr = util.calculate_psnr(img_Sr, img_Hr,)
        print(psnr)
        sum_psnr += psnr
        max_psnr = max(max_psnr,psnr)
        min_psnr = min(min_psnr, psnr)
        ssim = util.calculate_ssim(img_Sr, img_Hr,)
        # print(ssim)
        sum_ssim += ssim
        max_ssim = max(max_ssim, ssim)
        min_ssim = min(min_ssim, ssim)
    print('Average psnr = ', sum_psnr / numbers)
    print('min_psnr = ', min_psnr)
    print('Max_psnr = ', max_psnr)
    print('Average ssim = ', sum_ssim / numbers)
    print('min_ssim = ', min_ssim)
    print('Max_ssim = ', max_ssim)


def evulate_diff_name():
    hr_paths = util.get_image_paths(HR_path)
    numbers = len(hr_paths)
    sum_psnr = 0
    max_psnr = 0
    min_psnr = 100
    sum_ssim = 0
    max_ssim = 0
    min_ssim = 1
    for hr_path in tqdm(hr_paths):
        name, ext = os.path.splitext(os.path.basename(hr_path))
        # img_name = os.path.basename(hr_path)
        temp = str(name) + '.jpg'
        sr_path = os.path.join(SR_path, temp)

        img_Hr = util.imread_uint(hr_path, n_channels=n_channels)  # HR image, int8
        img_Sr = util.imread_uint(sr_path, n_channels=n_channels)  # HR image, int8

        psnr = util.calculate_psnr(img_Sr, img_Hr,)
        sum_psnr += psnr
        max_psnr = max(max_psnr, psnr)
        min_psnr = min(min_psnr, psnr)

        ssim = util.calculate_ssim(img_Sr, img_Hr,)
        sum_ssim += ssim
        max_ssim = max(max_ssim, ssim)
        min_ssim = min(min_ssim, ssim)
    print('Average psnr = ', sum_psnr / numbers)
    print('min_psnr = ', min_psnr)
    print('Max_psnr = ', max_psnr)
    print('Average ssim = ', sum_ssim / numbers)
    print('min_ssim = ', min_ssim)
    print('Max_ssim = ', max_ssim)


if __name__ == '__main__':
    print('-------------------------compute psnr and ssim for evaluate sr model---------------------------------')
    # evulate()
    HR_path = 'D:\Study\Codes\PSFR-GAN\datasets\FFHQ\imgs512'
    SR_path = 'D:\Study\Codes\PSFR-GAN\datasets\FFHQ\lrs512'
    n_channels = 3
    evulate_diff_name()



