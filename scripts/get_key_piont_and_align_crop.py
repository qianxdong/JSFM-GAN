#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：get_key_piont_and_align_crop.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/6/17 16:31 
'''

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging

from dete_faces.predict import dete_face
from skimage import transform as trans

logger = logging.getLogger(__name__)
logging.basicConfig(filename="test.log", level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def get_key_points(pic_path, savepath):
    '''
    @param pic_path: FFHQ IAMGES
    @param savepath: KEY POINTS.NPY SAVE ROOT
    @return: None
    '''
    pics = os.listdir(pic_path)
    key_points = []
    (x1, y1) = (0, 0)
    (x2, y2) = (0, 0)
    (x3, y3) = (0, 0)
    (x4, y4) = (0, 0)
    (x5, y5) = (0, 0)

    for index, pic in enumerate(tqdm(pics)):
        image = cv2.imread(os.path.join(pic_path, pic))
        _, _, lmk_points = dete_face(image)
        assert len(lmk_points) > 0, 'No faces'
        (x1, y1) = (lmk_points[0][0][0] + x1, lmk_points[0][0][1] + y1)
        (x2, y2) = (lmk_points[0][1][0] + x2, lmk_points[0][1][1] + y2)
        (x3, y3) = (lmk_points[0][2][0] + x3, lmk_points[0][2][1] + y3)
        (x4, y4) = (lmk_points[0][3][0] + x4, lmk_points[0][3][1] + y4)
        (x5, y5) = (lmk_points[0][4][0] + x5, lmk_points[0][4][1] + y5)

    if y1==y2:
        (x1, y1) = (x1 / len(pics), y1 / len(pics))
        (x2, y2) = (x2 / len(pics), y2 / len(pics))
        (x3, y3) = (x3 / len(pics), y3 / len(pics))
        (x4, y4) = (x4 / len(pics), y4 / len(pics))
        (x5, y5) = (x5 / len(pics), y5 / len(pics))
    else:
        y1 = (y1+y2)//2
        y2 = y1
        (x1, y1) = (x1 / len(pics), y1 / len(pics))
        (x2, y2) = (x2 / len(pics), y2 / len(pics))
        (x3, y3) = (x3 / len(pics), y3 / len(pics))
        (x4, y4) = (x4 / len(pics), y4 / len(pics))
        (x5, y5) = (x5 / len(pics), y5 / len(pics))
    key_points.append((x1, y1))
    key_points.append((x2, y2))
    key_points.append((x3, y3))
    key_points.append((x4, y4))
    key_points.append((x5, y5))

    np.save(f"{savepath}", key_points)


def align_crop(test_path, npy_path, save_path, scale=1, size_threshold=999):
    '''
    @param test_path: path of pics
    @param npy_path: key points of ffhq dataset
    @param save_path: save-path of align_crop images
    @param scale: modify the key points for the different size image
    @param size_threshold: threshold for divide faces in normal or big face
    @return: None
    '''

    os.makedirs(save_path, exist_ok=True)
    pics = os.listdir(test_path)
    align_out_size = (112/scale, 96/scale)
    ref_points = np.load(npy_path)/scale

    for index, pic in enumerate(tqdm(pics)):
        image = cv2.imread(os.path.join(test_path, pic))
        if image is None:
            logging.info(f"{pic} is dataloss")
            os.remove(os.path.join(test_path, pic))
            pass
        else:
            img, locations, lmk_points = dete_face(image)

            if len(locations) == 0:
                logging.info(f"{pic} No faces detected")
                pass

            for i, location in enumerate(locations):
                if abs(location[0][0]-location[1][0]) > size_threshold or abs(location[0][1]-location[1][1])>size_threshold:
                    logging.info(f"Face of {pic} No.{i} is too large")
                    pass
                single_points = []
                for j in range(5):
                    single_points.append([lmk_points[i][j][0], lmk_points[i][j][1]])
                single_points = np.array(single_points)
                tform = trans.SimilarityTransform()
                tform.estimate(single_points, ref_points)
                tmp_face = trans.warp(img, tform.inverse, output_shape=align_out_size, order=3)
                aligned_faces = tmp_face * 255
                cv2.imwrite(os.path.join(save_path, f'{index}_{i}.jpg'), aligned_faces)


def change_name(pic_path):
    pics = os.listdir(pic_path)
    for index, pic in enumerate(tqdm(pics)):
        os.rename(os.path.join(pic_path, pic), os.path.join(pic_path, f'{index+1}_{len(pics)}.jpg'))


if __name__ == '__main__':

    pics_path = 'D:\w\SR-TEST\HR\Test_CelebA-HQ-gt'
    key_point_root = r'D:\w\SR-TEST\SR\retinaface_key_points_y3.npy'
    if not os.path.exists(key_point_root):
        get_key_points(pics_path, key_point_root)
    # change_name(r'D:\w\SR-TEST\temp\face_align')
    align_crop(test_path=r'C:\Users\Administrator\Desktop\caoling',
               npy_path=key_point_root,
               scale=1,
               save_path=r'C:\Users\Administrator\Desktop\ghjg')
