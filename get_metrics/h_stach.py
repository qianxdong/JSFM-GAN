#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：h_stach.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/6/3 14:38 
'''


import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def Partial_magnification(pic, target, location='lower_right', ratio=2):
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_h]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''
    # img = copy.copy(pic)

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

def check_num():
    max_path = r'D:\w\SR-TEST\SR\GFPGAN_SR\CelebA-SR_GFPGAN_x4\restored_imgs'
    min_path = r'D:\w\SR-TEST\SR\GFPGAN_SR\CelebA-SR_GFPGAN_x4\restored_faces'
    save_path = r'D:\w\SR-TEST\SR\GFPGAN_SR\CelebA-SR_GFPGAN_x4\imgs'
    os.makedirs(save_path, exist_ok=True)

    err = 0
    min_imgs = os.listdir(min_path)
    for idx, name in enumerate(tqdm(min_imgs)):
        base = name.split('_')[0]+'.jpg'
        if os.path.exists(os.path.join(max_path, base)):
            img = cv2.imread(os.path.join(max_path, base))
            cv2.imwrite(os.path.join(save_path, base), img)
        else:
            err = err+1
    print(f'图片数量：{len(min_imgs)-err}，未处理：{err}')

def change_name(name,type_SR):
    if name == "DFDNet":
        name = f"{name}_SR/{type_SR}_{name}/Step4_FinalResults"
    elif name == "ESRGAN":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "SuperFAN":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "Deblurv2":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "HiFaceGAN":
        name = f"{name}_SR/{type_SR}_{name}/face_renov_2"

    elif name =='Pulse':
        name = f"{name}_SR/{type_SR}_{name}/hq"

    elif name == "GPEN":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "GFPGAN":
        name = f"{name}_SR/aligned_{type_SR}_{name}/restored_imgs"
        # name = f"{name}_SR/{type_SR}_{name}/restored_imgs"

    elif name == "Panini":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "PSFRGAN":
        name = f"{name}_SR/{type_SR}_{name}/hq"

    elif name == "PSFRGAN-FPN":
        name = f"{name}_SR/{type_SR}_{name}/hq"

    elif name == "my":
        name = f"{name}_SR/{type_SR}_{name}/hq"

    return name

def change_nameX4(name,type_SR):
    if name == "ESRGAN":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "SuperFAN":
        name = f"{name}_SR/{type_SR}_{name}"

    elif name == "GFPGAN":
        name = f"{name}_SR/aligned_{type_SR}_{name}_x4/restored_imgs"
    elif name == "Panini":
        name = f"{name}_SR/Test_CelebA-HQ_test-x4-512_result"

    elif name == "PSFRGAN":
        name = f"{name}_SR/{type_SR}_{name}_x4/hq"

    elif name == "my":
        name = f"{name}_SR/x4"

    return name


def random_gray(p=1):
    import imgaug.augmenters as iaa
    """input single RGB PIL Image instance"""
    x = cv2.imread(r'D:\w\SR-TEST/142.jpg')
    x = x[np.newaxis, :, :, :]
    aug = iaa.Sometimes(p, iaa.Grayscale(alpha=1.0))
    aug_img = aug(images=x)
    print(type(aug_img[0]),aug_img[0].shape)
    cv2.imwrite(r'D:\w\SR-TEST/gray.jpg',aug_img[0])
    # print(aug_img)




def is_valid_image(path):

    '''
    check .jpg image
    '''
    try:
        isValid = True
        fileObj = open(path, 'rb')  # open image with binary format
        buf = fileObj.read()
        if not buf.startswith(b'\xff\xd8'):  # start with '\xff\xd8'
            isValid = False
        elif buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):  # end with '\xff\xd9'
                isValid = False
        else:
            try:
                Image.open(fileObj).verify()
            except Exception as e:
                isValid = False
    except Exception as e:
        print('Can\'t open file: {}'.format(path))
        return

    if not isValid:
        print('{} is damaged.'.format(path))
    return

# if __name__ == '__main__':
#         dataPath = "/home/tang/work/QXD/PSFR-GAN/dataset/FFHQ_10000/imgs512_10000/"
#         imgList = os.listdir(dataPath)
#         for img in tqdm(imgList):
#             path = os.path.join(dataPath, img)
#             is_valid_image(path)

def selct():
    imgpath = r'D:\w\SR-TEST\compare\CelebA_SRCelebA_SR'
    savepath = r'D:\w\SR-TEST\compare\x4mouth'
    os.makedirs(savepath, exist_ok=True)
    # idx = [10,50,107,76,136,339,610,722,738,805,921,982,1876,1979,7654,29,1328,1241,1154] #x4left
    idx = [116,212,244,340,481,663,723,980,1077,1189,1217,1240,1271,1309,1389,1653,1672,1722,1739,1740,1746,1768,1924]#x4mouth
    # idx = [76,192,206,218,242,249,179,280,328,334,345,440,482,496,498,513,538,554,558,615,618,633,648,673,722,736,771,805,905,
    #        921,969,998,1024,1057,1058,1087,1091,1125,1127,1194,1497,1501,1506,1572,1632,1666,1686,1727,1818,1926,1950,1963] #tleft

    # idx = [40,66,116,263,340,489,496,586,723,980,1187,1208,1389,1447,1448,1614,1667,1740,1746,1924]#tmouth
    # idx = [2,27,30,33,38,45,53,62,65,70,157,213,266,321,336,339,395,407,451,462,550,551,578,983,1026,1031]#online right
    # idx = [57,163,251,321,347,548,553,582,650,732,943,945,1048,1054]#online mouth

    imgs = os.listdir(imgpath)
    for i, name in enumerate(tqdm(imgs)):
        try:
            # if int(name.split('_1056.')[0]) in idx:
            if int(name.split('.')[0]) in idx:
                tmp = cv2.imread(os.path.join(imgpath, name))
                cv2.imwrite(os.path.join(savepath,name),tmp)
        except:
                pass

def protin():
    picpath = r'D:\w\protin'
    savepath ='D:\w\protin_big'
    os.makedirs(savepath,exist_ok=True)
    target = [[295,300, 387-295, 400-300], 'lower_left', 2]
    pics = os.listdir(picpath)
    for i,name in enumerate(tqdm(pics)):
        lr = cv2.imread(os.path.join(picpath, name))
        lr = Partial_magnification(pic=lr, target=target[0], location=target[1], ratio=target[2])  #lower_right,lower_left,top_right,top_left,center
        cv2.imwrite(os.path.join(savepath,name),lr)

# protin()

def pro_base_to():
    import natsort
    # target = [[120, 193, 250 - 120, 280 - 193], 'lower_right', 2]  # left eye
    # target = [[279, 187, 389 - 279, 271 - 187], 'lower_left', 2]   # right eye
    # target = [[168, 340, 341 - 168, 425 - 340], 'top_left', 2]    # mouth
    target = [[286, 266, 402 - 286, 378 - 268], 'top_left', 2]  # mouth
    files_name= [r'D:\w\SR-TEST\LR\Test_CelebA-HQ_test',r'D:\w\tmp_pic\baseline',r'D:\w\tmp_pic\baseline_aub\hq',r'D:\w\tmp_pic\baseline_jsfm',r'D:\w\tmp_pic\baseline_jsfm_stage',r'D:\w\SR-TEST\SR\my_SR\CelebA_SR_my\hq']
    LR_PIC_path = r'D:\w\SR-TEST\LR\Test_CelebA-HQ_test'
    savepath = r'D:\w\tmp_pic\conpare_Ablation_other'
    os.makedirs(savepath,exist_ok=True)
    white = np.ones((512, 30, 3), dtype=np.int8) * 255
    # LR_pics = os.listdir(LR_PIC_path)
    LR_pics = natsort.natsorted(os.listdir(LR_PIC_path), alg=natsort.ns.PATH)
    for j, name in enumerate(tqdm(LR_pics)):
        if j in [107,328,342,498,601,650,771,864,896,1027,1091,1135,1234,1309,1382,1504,1653,1687,1694,1776,1822]:
            lr = cv2.imread(os.path.join(LR_PIC_path, name))
            sr = np.zeros_like(lr)
            for i, file_name in enumerate(files_name):
                if not os.path.exists(os.path.join(file_name, name)):
                        temp = np.ones_like(lr) * 255
                else:
                    temp = cv2.imread(os.path.join(file_name, name))
                    temp = Partial_magnification(pic=temp, target=target[0], location=target[1], ratio=target[2])
                if i == 0:
                    sr = temp
                else:
                    sr = np.hstack((sr, white, temp))
            cv2.imwrite(os.path.join(savepath,name),sr)
        else:
            pass
# pro_base_to()

def changesize():
    pic = r'D:\w\tmp_pic\cam/individualImage.png'
    savepath = r'D:\w\tmp_pic\cam'
    sr = 'sr'
    lr = 'lr'
    os.makedirs(savepath, exist_ok=True)
    temp = cv2.imread(pic)
    # print(temp.shape)
    # LR = temp[:,0:512, :]
    # for i in [512,256, 128, 64, 32,16]:
    #     if i ==512:
    #         cv2.imwrite(os.path.join(savepath, lr + str(i) + '.png'),LR)
    #     else:
    #         cv2.imwrite(os.path.join(savepath, lr+str(i)+'.png'), cv2.resize(cv2.resize(LR,(i,i)),(512,512)))
    # for i in range(3, 6):
    #     SR = temp[:, 512*i:512*(i+1), :]
    #     size = 2 ** (i+3)
    #     print(size)
    #     cv2.imwrite(os.path.join(savepath, sr+str(size)+'.png'), SR)
    #     if i == 5:
    #         cv2.imwrite(os.path.join(savepath, sr + str(512) + '.png'), cv2.resize(SR,(512,512)))
    HR = temp[:, 512*2:512*(2+1), :]
    LR = temp[:, 512 * 0:512 * (0 + 1), :]
    D = HR-LR
    cv2.imshow('HR', HR)
    cv2.imshow('LR', LR)
    cv2.imshow('D' , D)
    cv2.imshow('T' , (D + LR))
    cv2.waitKey(0)
    # for i in [32,16]:
    #     cv2.imwrite(os.path.join(savepath, sr+str(i)+'.png'), cv2.resize(cv2.resize(SR,(i,i)),(512,512)))
# changesize()
if __name__ == '__main__':
    # check_num()
    # selct()
    # random_gray()

    x4 = True
    choose = 0 # [CelebA_SR:0, 1:Lfw_SR, 2:Online_SR]
    target = [[120, 193, 250 - 120, 280 - 193], 'lower_right', 2]  # left eye
    # target = [[279, 187, 389 - 279, 271 - 187], 'lower_left', 2]   # right eye
    # target = [[168, 340, 341 - 168, 425 - 340], 'top_left', 2]    # mouth

    gt_path = 'Test_CelebA-HQ-gt'
    base_path = r'D:\w\SR-TEST'
    types = ["LR", "SR", "HR"]
    data_types = ['Test_CelebA-HQ_test', 'Test_lfw_test', 'Test_online_test_align']
    if x4:
        files_name = ["ESRGAN", "SuperFAN","Panini", "GFPGAN", "PSFRGAN", "my"]
    else:
        files_name = ["Deblurv2", "HiFaceGAN", 'Pulse', "GPEN", "GFPGAN", "Panini", "DFDNet", "PSFRGAN", "my"]

    save_types = ['CelebA_SR', 'Lfw_SR', 'Online_SR']

    LR_path = os.path.join(base_path, types[0])
    # LR_path = 'D:\w\SR-TEST\LR\Test_CelebA-HQ_test-x4-128'
    SR_path = os.path.join(base_path, types[1])
    HR_path = os.path.join(base_path, types[2])
    white = np.ones((512, 30, 3), dtype=np.int8) * 255
    if choose == 0:
        LR_PIC_path = os.path.join(LR_path, data_types[choose])
        LR_pics = os.listdir(LR_PIC_path)

        # LR_pics = os.listdir(LR_path)
        if x4:
            SR_PIC_path = os.path.join(SR_path, change_nameX4(files_name[0], save_types[choose]))
        else:
            SR_PIC_path = os.path.join(SR_path, change_name(files_name[0], save_types[choose]))

        SR_pics = os.listdir(SR_PIC_path)

        HR_PIC_path = os.path.join(HR_path, gt_path)
        HR_pics = os.listdir(HR_PIC_path)

        for j, name in enumerate(tqdm(LR_pics)):
            lr = cv2.imread(os.path.join(LR_PIC_path, name))
            # lr = cv2.resize(cv2.imread(os.path.join(LR_path, name)),(512,512))
            lr = Partial_magnification(pic=lr, target=target[0], location=target[1], ratio=target[2])  #lower_right,lower_left,top_right,top_left,center
            sr = np.zeros_like(lr)
            for i, file_name in enumerate(files_name):
                if x4:
                    if not os.path.exists(os.path.join(SR_path, change_nameX4(file_name, save_types[choose]), name)):
                        temp = np.ones_like(lr) * 255
                    else:
                        temp = cv2.imread(os.path.join(SR_path, change_nameX4(file_name, save_types[choose]), name))
                        temp = Partial_magnification(pic=temp, target=target[0], location=target[1], ratio=target[2])
                else:
                    if not os.path.exists(os.path.join(SR_path, change_name(file_name, save_types[choose]), name)):
                        if not os.path.exists(os.path.join(SR_path, change_name(file_name, save_types[choose]), name.split('.')[0]+'.png')):
                            temp = np.ones_like(lr) * 255
                        else:
                            temp = cv2.imread(os.path.join(SR_path, change_name(file_name, save_types[choose]), name.split('.')[0]+'.png'))
                            temp = Partial_magnification(pic=temp, target=target[0], location=target[1],
                                                         ratio=target[2])
                    else:
                        temp = cv2.imread(os.path.join(SR_path, change_name(file_name, save_types[choose]), name))
                        temp = Partial_magnification(pic=temp, target=target[0], location=target[1], ratio=target[2])
                if i == 0:
                    sr = temp
                else:
                    sr = np.hstack((sr, white, temp))

            hr = cv2.imread(os.path.join(HR_PIC_path, name))
            hr = Partial_magnification(pic=hr, target=target[0], location=target[1], ratio=target[2])
            compare = np.hstack((lr, white, sr, white, hr))
            os.makedirs(f'D:\w\SR-TEST\compare/{save_types[choose]}{save_types[choose]}', exist_ok=True)
            cv2.imwrite(f'D:\w\SR-TEST\compare/{save_types[choose]}{save_types[choose]}/{name}', compare)
            # "LR","ESRGAN", "SuperFAN", "Deblurv2","HiFaceGAN", "DFDNet",  "GPEN", "GFPGAN", "PSFRGAN", "PSFRGAN-FPN", "GT"
    else:
        LR_PIC_path = os.path.join(LR_path, data_types[choose])
        LR_pics = os.listdir(LR_PIC_path)

        for j, name in enumerate(tqdm(LR_pics)):
            lr = cv2.imread(os.path.join(LR_PIC_path, name))
            lr = Partial_magnification(pic=lr, target=target[0], location=target[1], ratio=target[2])
            sr = np.zeros_like(lr)
            for i, file_name in enumerate(files_name):
                if x4:
                    if not os.path.exists(os.path.join(SR_path, change_nameX4(file_name, save_types[choose]), name)):
                        temp = np.ones_like(lr) * 255
                    else:
                        temp = cv2.imread(os.path.join(SR_path, change_nameX4(file_name, save_types[choose]), name))
                        temp = Partial_magnification(pic=temp, target=target[0], location=target[1], ratio=target[2])
                else:
                    if not os.path.exists(os.path.join(SR_path, change_name(file_name, save_types[choose]), name)):
                        if not os.path.exists(os.path.join(SR_path, change_name(file_name, save_types[choose]),
                                                           name.split('.')[0] + '.png')):
                            temp = np.ones_like(lr) * 255
                        else:
                            temp = cv2.imread(os.path.join(SR_path, change_name(file_name, save_types[choose]),
                                                           name.split('.')[0] + '.png'))
                            temp = Partial_magnification(pic=temp, target=target[0], location=target[1],
                                                         ratio=target[2])
                    else:
                        temp = cv2.imread(os.path.join(SR_path, change_name(file_name, save_types[choose]), name))
                        temp = Partial_magnification(pic=temp, target=target[0], location=target[1], ratio=target[2])
                if i == 0:
                    sr = temp
                else:
                    sr = np.hstack((sr, white, temp))

            compare = np.hstack((lr,white, sr))
            os.makedirs(f'D:\w\SR-TEST\compare/{save_types[choose]}{save_types[choose]}', exist_ok=True)
            cv2.imwrite(f'D:\w\SR-TEST\compare/{save_types[choose]}{save_types[choose]}/{name}', compare)
            # "LR", "ESRGAN", "SuperFAN", "Deblurv2","HiFaceGAN", "DFDNet",  "GPEN", "GFPGAN", "PSFRGAN", "PSFRGAN-FPN"

    #
    # selct()
    #
    #
