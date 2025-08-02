#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    @File    ：down_mixup.py
@Author  ：Xiaodong Qian
@Function:image degradation
@Date    ：2022/3/18 21:26 
'''

import os
from tqdm import tqdm
from PIL import Image
import argparse

def down_size(imag, size, type=Image.ANTIALIAS):
    """
    PIL.Image.NEAREST（使用最近的临近）
    PIL.Image.BILINEAR（线性插值）
    PIL.Image.BICUBIC（三次样条插值）
    PIL.Image.LANCZOS（高质量的下采样滤波器）
    # Image.ANTIALIAS
    """
    down_pic = imag.resize(size, type)
    return down_pic


def mix_up(im1, im2, alpha):
    im1 = Image.open(im1)
    im2 = Image.open(im2)
    size2 = im2.size
    im11 = im1.resize(size2)
    mix_image = Image.blend(im11, im2, alpha)
    return mix_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加命令行参数
    parser.add_argument("--mode", type=str, default="resize", help="resize or mix")
    parser.add_argument("--size", type=int, default=512, help="图像大小")
    parser.add_argument("--pic_dir", type=str, default="pics", help="图片文件夹")
    parser.add_argument("--save_dir", type=str, default="output", help="图片保存文件夹")


    # 解析命令行参数
    args = parser.parse_args()
    mode = args.mode
    size = (args.size, args.size)
    pic_dir = args.pic_dir
    save_dir = args.save_dir


    # 输出解析后的参数
    print("Input file:", args.input)
    print("Output file:", args.output)
    if mode == 'resize':
        try:
            pics = os.listdir(pic_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f'make the {save_dir} successful!')
            else:
                print(f'the {save_dir} existed!')
        except:
            raise AssertionError("The pics file is empty!")

        for pic in tqdm(pics):
            img = Image.open(os.path.join(pic_dir, pic))
            if img is None:
                print(f'{pic} height is wrong !')
                continue
            img = down_size(img, size)
            img.save(os.path.join(save_dir, pic.split('.')[0]+'.jpg'))
        print(f'改变尺寸至{size}结束,保存至{save_dir}')

    elif mode == 'mix':
        try:
            pics = os.listdir(pic_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f'make the {save_dir} successful!')
            else:
                print(f'the {save_dir} existed!')
        except:
            raise AssertionError("The pics file is empty!")

        im1 = ''
        im2 = ''
        for i, pic in enumerate(tqdm(pics)):
            if i % 2 == 0:
                im1 = os.path.join(pic_dir, pic)
            else:
                im2 = os.path.join(pic_dir, pic)
            if im1 != '' and im2 != '':
                mix_image = mix_up(im1, im2, 0.1)
                mix_image.save(os.path.join(save_dir, f'{i}.jpg'))
                im1 = ''
                im2 = ''
    else:
        raise ValueError('PLEASE CHOOSE THE RIGHT MODE!')
