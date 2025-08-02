#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File     ：image_degradation.py
@Function :image degradation
@Statement:Code reference PSFR-GAN and Face-Renovation
'''

import cv2
import os
from tqdm import tqdm
import numpy as np
import imgaug.augmenters as ia
import imgaug as iaa
from PIL import Image
import argparse


def get_down():
    scale_size = np.random.randint(32, 256)
    return ia.Sequential([
        ia.Resize(scale_size, interpolation=iaa.ALL),
        ia.Resize({"height": 512, "width": 512})
    ])


def get_noise():
    return ia.OneOf([
        ia.AdditiveGaussianNoise(scale=(20, 25), per_channel=True),
        ia.AdditiveLaplaceNoise(scale=(20, 25), per_channel=True),
        ia.AdditivePoissonNoise(lam=(15, 25), per_channel=True),
    ])


def get_blur():
    return ia.Sequential([
        ia.Sometimes(0.5, ia.OneOf([ia.GaussianBlur((3, 15))])),
        ia.Sometimes(0.5, ia.OneOf([ia.MotionBlur((5, 25))])),
        ia.AverageBlur(k=(3, 15)),
        ia.MedianBlur(k=(3, 15)),
        ])


def get_jpeg():
    return ia.JpegCompression(compression=(10, 65))


def get_all():
    return ia.Sequential([
        get_blur(),
        get_down(),
        get_noise(),
        get_jpeg(),
    ])


def get_by_suffix(de_type):
    if de_type == 'down':
        return get_down()
    elif de_type == 'noise':
        return get_noise()
    elif de_type == 'blur':
        return get_blur()
    elif de_type == 'jpeg':
        return get_jpeg()
    elif de_type == 'blind':
        return get_all()
    else:
        raise ValueError(f'{de_type} not supported, please choose the right de_type!')


def create_mixed_dataset(input_dir, de_type='blind', save_type='lr'):
    output_dir = input_dir + '_' + de_type
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    trans = get_by_suffix(de_type)
    mix_degrade = lambda x: trans.augment_image(x)
    for item in tqdm(os.listdir(input_dir)):
        hr = cv2.imread(os.path.join(input_dir, item))
        lr = mix_degrade(hr)
        if save_type == 'lr':
            cv2.imwrite(os.path.join(output_dir, item.split('.')[0]+'.jpg'), lr)
        else:
            img = np.concatenate((lr, hr), axis=0)
            cv2.imwrite(os.path.join(output_dir, item.split('.')[0]+'.jpg'), img)

# ============================================================================================

def random_gray(x, p=0.3):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug = ia.Sometimes(p, ia.Grayscale(alpha=1.0))
    aug_img = aug(images=x)
    return aug_img[0]


def complex_imgaug(x, org_size, scale_size):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug_seq = ia.Sequential([
        ia.Sometimes(0.5, ia.OneOf([
            ia.GaussianBlur((3, 15)),
            ia.AverageBlur(k=(3, 15)),
            ia.MedianBlur(k=(3, 15)),
            ia.MotionBlur((5, 25))
        ])),
        ia.Resize(scale_size, interpolation=iaa.ALL),
        ia.Sometimes(0.2, ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)),
        ia.Sometimes(0.7, ia.JpegCompression(compression=(10, 65))),
        ia.Resize(org_size),
    ])
    aug_img = aug_seq(images=x)
    return aug_img[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--mode", type=str, default="resize", help="down/noise/blur/jpeg/blind/debug")
    parser.add_argument("--size", type=int, default=512, help="图像大小")
    parser.add_argument("--pic_dir", type=str, default="pics", help="图片文件夹")
    parser.add_argument("--save_dir", type=str, default="save", help="图片保存文件夹")

    parser.add_argument("--gray_proportion", type=float, default=0.3, help="灰度化比例")
    parser.add_argument("--scale1", type=int, default=32, help="尺度下限")
    parser.add_argument("--scale2", type=int, default=256, help="尺度上限")
    # 解析命令行参数
    args = parser.parse_args()

    mode = args.mode
    hr_size = args.size
    img_path = args.pic_dir
    sava_path = args.save_dir

    # 单一退化可选择
    # create_mixed_dataset(img_path, mode)

    if not os.path.exists(sava_path):
        os.makedirs(sava_path)

    for indix,pic in enumerate(tqdm(os.listdir(img_path))):
        if mode == 'debug':
            if indix>1:
                break
        # 读取图像
        hr_img = Image.open(os.path.join(img_path, pic)).convert('RGB')

        # resize
        hr_img = hr_img.resize((hr_size, hr_size))
        # scale
        scale_size = np.random.randint(32, 256)
        # gray
        if mode == 'gray':
            lr_img = random_gray(hr_img, p=args.gray_proportion)
            Image.fromarray(lr_img).save(os.path.join(sava_path, pic))
        elif mode == 'blind':
            lr_img = complex_imgaug(hr_img, hr_size, scale_size)
            Image.fromarray(lr_img).save(os.path.join(sava_path, pic))

