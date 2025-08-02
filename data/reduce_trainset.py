#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：reduce_trainset.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/7/17 11:01 
'''

import os
import shutil
from tqdm import tqdm
import argparse


def copy_pics(images_path, target_path, num, *args):
    images = os.listdir(images_path)
    assert num < len(images), 'Expect num smaller than items of images file!'
    os.makedirs(target_path, exist_ok=True)
    for i, pic in enumerate(tqdm(images, total=num)):
        if i == num:
            break
        if not os.path.exists(os.path.join(target_path, pic)):
            pic_path = os.path.join(images_path, pic)
            shutil.copy(pic_path, target_path)
        else:
            pass
    print(f'Copy {num} pics from <<{images_path}>> to <<{target_path}>> over!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='参数解析')
    parser.add_argument('source_dir', type=str, help='The path to the source directory containing images.')
    parser.add_argument('target_dir', type=str, help='The path to the target directory where images will be copied.')
    parser.add_argument('num_images', type=int, help='The number of images to copy.')
    args = parser.parse_args()
    copy_pics(args.source_dir, args.target_dir, args.num_images)
