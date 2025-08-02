#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：eva.py
@Author  ：Xiaodong Qian
@Function:Calculated metrics
@Date    ：2022/3/31 14:42 
'''
import os
cuda = ''
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import cv2
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
# from sklearn.metrics.pairwise import cosine_similarity
from imageio import imread
import numpy as np
from torchvision import transforms
import torch
# from memory_profiler import LineProfiler, profile

# import get_metrics.utils_image
import utils_image
# import pytorch_fid
import natsort
import lpips

from tqdm import tqdm
import csv
# from get_metrics.niqe import niqe
from niqe import niqe
from PIL import Image

# from get_metrics.FID import *
from FID import *

loss_fn_alex = lpips.LPIPS(net='alex')  # 可选择vgg,alex

from pytorch_msssim import ms_ssim


# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)


class PILToTensor(object):
    """Converts a PIL Image to a PyTorch tensor."""
    def __call__(self, pic):
        # Convert PIL image to NumPy array
        img = np.array(pic, dtype=np.float32)

        # Transpose image data from (H x W x C) to (C x H x W)
        img = img.transpose((2, 0, 1))

        # Convert NumPy array to PyTorch tensor
        tensor = torch.from_numpy(img).contiguous()

        return tensor


def get_ms_ssim(sr_path, gt_path):
    try:
        pilimag = transforms.PILToTensor()
    except:
        pilimag = PILToTensor()

    x = Image.open(sr_path)
    x = pilimag(x)
    X = torch.unsqueeze(x, dim=0).type(torch.float32)
    del x

    y = Image.open(gt_path)
    y = pilimag(y)
    Y = torch.unsqueeze(y, dim=0).type(torch.float32)
    del y

    m_ssim = ms_ssim(X, Y, data_range=255, size_average=False).item()
    return m_ssim


def get_ssim_psnr(im1, im2):
    img1 = imread(im1)
    img2 = imread(im2)
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    ssim = SSIM(img1, img2, multichannel=True)
    psnr = PSNR(img1, img2)
    # print(ssim, psnr)
    del img1, img2
    return ssim, psnr


# @profile
def get_lpips(im_path, gt_path):
    ig = utils_image.imread_uint(im_path, 3)
    gt = utils_image.imread_uint(gt_path, 3)
    input = utils_image.uint2tensor3(ig)
    gt = utils_image.uint2tensor3(gt)

    # ig = get_metrics.utils_image.imread_uint(im_path, 3)
    # gt = get_metrics.utils_image.imread_uint(gt_path, 3)
    # input = get_metrics.utils_image.uint2tensor3(ig)
    # gt = get_metrics.utils_image.uint2tensor3(gt)
    if gt.shape != input.shape:
        raise ValueError('The shape of gt do not match the images!')
    loss = loss_fn_alex(input, gt).item()
    del ig, gt, input
    return loss


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = Image.open(image1).convert('L')
    image2 = Image.open(image2).convert('L')
    # image1 = Image.open(image1)
    # image2 = Image.open(image2)
    x_1 = np.array(image1).reshape(1, -1)
    x_2 = np.array(image2).reshape(1, -1)
    con_sim = cosine_similarity(x_1, x_2).item()
    return con_sim


# python -m pytorch_fid path/to/dataset1 path/to/dataset2
# python -m pytorch_fid D:\Study\Codes\PSFR-GAN\datasets\FFHQ+\Face_full  D:\Study\Codes\PSFR-GAN\datasets\FFHQ+\Face --dims 2048
''''''

# def comm(image_path, gt_path):
#     count = 0
#     err = 0
#     psnr = 0
#     ssim = 0
#     lpips = 0
#     Niqe = 0
#     images = natsort.natsorted(os.listdir(image_path), alg=natsort.ns.PATH)
#     gts = natsort.natsorted(os.listdir(gt_path), alg=natsort.ns.PATH)
#     k = 4
#     save_item = [[], [], [], [], [],[]]
#     for i in range(k):
#         for idx, pic in enumerate(tqdm(images[i*len(images)//k:(i+1)*len(images)//k:1])):
#             # print(idx+i*len(images)//k, pic)
#             if pic == gts[idx+i*len(images)//k]:
#                 s, p = get_ssim_psnr(os.path.join(image_path, pic), os.path.join(gt_path, pic))
#                 lp = get_lpips(os.path.join(image_path, pic), os.path.join(gt_path, pic))
#                 pic_path = os.path.join(image_path, pic)
#                 niq = niqe(np.array(Image.open(pic_path).convert('LA'))[:, :, 0])
#                 print(type(niq))
#
#                 save_item[0].append(pic)
#                 save_item[1].append(round(s, 4))
#                 save_item[2].append(round(p, 4))
#                 save_item[3].append(round(float(lp), 4))
#                 save_item[4].append(round(float(niq), 4))
#                 ssim = ssim + s
#                 psnr = psnr + p
#                 lpips = lpips + lp
#                 Niqe = Niqe + niq
#                 count = count + 1
#                 del lp, s, p, niq
#             else:
#                 err = err + 1
#                 pass
#
#         ssim = ssim / (count + 10e-8)  # 避免分母为零
#         psnr = psnr / (count + 10e-8)
#         lpips = lpips / (count + 10e-8)
#         Niqe = Niqe / (count + 10e-8)
#
#     mssim = ssim / (count + 10e-8)  # 避免分母为零
#     mpsnr = psnr / (count + 10e-8)
#     lpips = lpips / (count + 10e-8)
#     Niqe = Niqe / (count + 10e-8)
#
#     return mssim, mpsnr, lpips, Niqe, err, save_item

def test(image_path, gt_path):
    images = natsort.natsorted(os.listdir(image_path), alg=natsort.ns.PATH)
    gts = natsort.natsorted(os.listdir(gt_path), alg=natsort.ns.PATH)
    count = 0
    err = 0
    psnr = 0
    ssim = 0
    lpips = 0
    Niqe = 0
    cos_similarity = 0

    for idx, pic in enumerate(images):
        if pic == gts[idx]:
            s, p = get_ssim_psnr(os.path.join(image_path, pic), os.path.join(gt_path, pic))
            lp = get_lpips(os.path.join(image_path, pic), os.path.join(gt_path, pic))
            pic_path = os.path.join(image_path, pic)
            niq = niqe(np.array(Image.open(pic_path).convert('LA'))[:, :, 0]).item()
            similarity = image_similarity_vectors_via_numpy(os.path.join(image_path, pic), os.path.join(gt_path, pic))
            ssim = ssim + s
            psnr = psnr + p
            lpips = lpips + lp
            Niqe = Niqe + niq
            cos_similarity = cos_similarity + similarity
            count = count + 1
            del lp, s, p, niq
        else:
            err = err + 1
            pass
    mssim = ssim / (count + 10e-8)  # 避免分母为零
    mpsnr = psnr / (count + 10e-8)
    lpips = lpips / (count + 10e-8)
    Niqe = Niqe / (count + 10e-8)
    cos_similarity = cos_similarity / (count + 10e-8)

    return mssim, mpsnr, lpips, Niqe, cos_similarity, err

def eaely_fid():
    value = []
    img_files = r'D:\w\total\SR\without'
    # img_files = r'D:\w\total\SR\FID'
    # img_files = r'D:\w\total\SR\without'
    gts_path = r'D:\w\SR-TEST\HR\Test_CelebA-HQ-gt'
    img_dir = natsort.natsorted(os.listdir(img_files), alg=natsort.ns.PATH)
    for idx, name_dir in enumerate(img_dir):
        # image_path = os.path.join(img_files, name_dir, 'hq')
        image_path = os.path.join(img_files, name_dir, 'hq')
        fid_value = calculate_fid_given_paths(paths=[image_path, gts_path], batch_size=50, cuda='', dims=2048)
        value.append(round(fid_value, 4))

    print(value)


# 79.2047 30.334740523142244, 28.24233007046209, 28.00532986664885, 24.829701601392344, 25.616515224524335, 25.87382366360032, 23.681161048561478, 23.7926467947278,   24.063258009143254, 23.31134452665333] [22.368111161134124, 22.486224083414754, 22.511285412219024, 22.871566284286246, 23.5198346257462
# 79.2047 216.10231946891102, 72.92265523863625, 46.51093656924678, 35.273993367040360, 32.12375097156729,  33.10116268212596, 29.593437149487926, 28.540586455992553, 27.57611405025787,  26.314389604174494] 30.098293154544876, 26.56361703148218,  25.51256143209713,  25.07556599277737,  24.4404437645683


def main():
    srnet = 'mm_SR'  #PSFRGAN-O_SR, PSFRGAN-FPN_SR, PSFRGAN_SR,GFPGAN_SR,DFDNet_SR,ESRGAN_SR,SuperFAN_SR,my_SR,Metrics_LR,HiFaceGAN_SR,Deblurv2_SR,GPEN_SR
    model_name = 'CelebA-LR'  # CelebA-LR,Online-LR,Lfw-LR,CelebA-LR-x4
    image_path = r'D:\Study\Codes\Compare_models\DMDNet\TestExamples\Results_TestGenericRestoration_02-25_17-43/'
    print(image_path)
    gt_path = r'D:\Study\Codes\PSFR-GAN\mydata\HR\Test_CelebA-HQ-gt'  # D:\Study\Codes\PSFR-GAN\mydata\HR\Test_CelebA-HQ-gt
    gts_path = r'D:\Study\Codes\PSFR-GAN\mydata\HR\Test_CelebA-HQ-gt/'
    images = natsort.natsorted(os.listdir(image_path), alg=natsort.ns.PATH)
    if gt_path != '':
        gts = natsort.natsorted(os.listdir(gt_path), alg=natsort.ns.PATH)
    count = 0
    err = 0
    psnr = 0
    ssim = 0
    lpips = 0
    Niqe = 0
    # cos_similarity = 0
    MS_SSIM = 0
    LF = 0
    save_item = [[], [], [], [], [], [], []]
    if gt_path != '':
        for idx, pic in enumerate(tqdm(images)):
            # if pic.split('_')[0]+'jpg' in gts:
            # lf = utils_image.lfd(os.path.join(image_path, pic), os.path.join(gt_path, pic))
            s, p = get_ssim_psnr(os.path.join(image_path, pic), os.path.join(gt_path, pic.split('.')[0]+'.jpg'))
            ms = get_ms_ssim(os.path.join(image_path, pic), os.path.join(gt_path, pic.split('.')[0]+'.jpg'))
            lp = get_lpips(os.path.join(image_path, pic), os.path.join(gt_path, pic.split('.')[0]+'.jpg'))

            # lf = utils_image.lfd(os.path.join(image_path, pic), os.path.join(gt_path, gts[idx]))
            lf = 0
            # s, p = get_ssim_psnr(os.path.join(image_path, pic), os.path.join(gt_path, gts[idx]))
            # ms = get_ms_ssim(os.path.join(image_path, pic), os.path.join(gt_path, gts[idx]))
            # lp = get_lpips(os.path.join(image_path, pic), os.path.join(gt_path, gts[idx]))

            pic_path = os.path.join(image_path, pic)
            niq = niqe(np.array(Image.open(pic_path).convert('LA'))[:, :, 0]).item()
            # similarity = image_similarity_vectors_via_numpy(os.path.join(image_path, pic), os.path.join(gt_path, pic))
            save_item[0].append(pic)
            save_item[1].append(round(s, 4))
            save_item[2].append(round(p, 4))
            save_item[3].append(round(lp, 4))
            save_item[4].append(round(niq, 4))
            save_item[5].append(round(ms, 4))
            save_item[6].append(round(lf, 4))
            ssim = ssim + s
            psnr = psnr + p
            lpips = lpips + lp
            Niqe = Niqe + niq
            MS_SSIM = MS_SSIM + ms
            LF = LF + lf
            # cos_similarity = cos_similarity + similarity
            count = count + 1
            del lp, s, p, niq, lf
        # else:
        #     err = err + 1
        #     pass
    else:
        for idx, pic in enumerate(tqdm(images)):
            pic_path = os.path.join(image_path, pic)
            niq = niqe(np.array(Image.open(pic_path).convert('LA'))[:, :, 0]).item()
            save_item[4].append(round(niq, 4))
            Niqe = Niqe + niq
            count = count + 1

    mssim = ssim / (count + 10e-8)  # 避免分母为零
    mpsnr = psnr / (count + 10e-8)
    lpips = lpips / (count + 10e-8)
    Niqe = Niqe / (count + 10e-8)
    MS_SSIM = MS_SSIM / (count + 10e-8)
    LFD = LF / (count + 10e-8)
    fid_value = calculate_fid_given_paths(paths=[image_path, gts_path], batch_size=50, cuda=cuda, dims=2048)
    # with open(f"D:\w\SR-TEST\SR/{srnet}/metrics_{model_name}.txt", "w") as f:  # 打开文件
    #     info = f'The pic path is {image_path}\nThe gt path is {gt_path}\n' \
    #            f' Tips: `|` means the bigger the better, .|. means the smaller the better\n' \
    #            f'    `|`ssim    = {round(mssim, 4)}\n' \
    #            f'    `|`psnr    = {round(mpsnr, 4)}\n' \
    #            f'    .|.lpsips  = {round(lpips, 4)}\n' \
    #            f'    .|.niqe    = {round(Niqe, 4)}\n' \
    #            f'    `|`MS_SSIM = {round(MS_SSIM, 4)}\n' \
    #            f'    .|.LFD = {round(LFD, 4)}\n' \
    #            f'    .|.FID     = {round(fid_value, 4)}'
    #     f.write(info)
    # f.close()
    #
    # f = open(f'D:\w\SR-TEST\SR/{srnet}/metrics_{model_name}.csv', 'w', newline='', encoding='utf-8')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["ID", "Name", "SSIM", "PSNR", "LPIPS", "NIQE", "MS_SSIM"])  # 构建列表头
    # if len(save_item[0])==0 and len(save_item[1])==0 and len(save_item[2])==0 and len(save_item[0])==0 and len(save_item[5])==0:
    #     for i in tqdm(range(len(save_item[4]))):
    #         csv_writer.writerow([i, 0, 0, 0, 0, save_item[4][i], 0])
    # else:
    #     for i in tqdm(range(len(save_item[4]))):
    #         csv_writer.writerow([i, save_item[0][i], save_item[1][i], save_item[2][i], save_item[3][i], \
    #                              save_item[4][i], save_item[5][i], save_item[5][i]])
    # f.close()
    if err > 0:
        print(f'err = {err}')
    print(f'The pic path is {image_path}\nThe gt path is {gt_path}\n'
          f'mssim   = {round(mssim, 4)}\n'
          f'mpsnr   = {round(mpsnr, 4)}\n'
          f'lpips   = {round(lpips, 4)}\n'
          f'niqe    = {round(Niqe, 4)}\n'
          f'MS_SSIM = {round(MS_SSIM, 4)}\n'
          f'LFD     = {round(LFD, 4)}\n'
          f'FID     = {round(fid_value, 4)}')

def change_name():
    pics_path = r'D:\w\SR-TEST\SR\GFPGAN_SR\aligned_CelebA_SR_GFPGAN_x4\restored_faces'
    save_path = r'D:\w\SR-TEST\SR\GFPGAN_SR\aligned_CelebA_SR_GFPGAN_x4\restored_imgs'
    pics = os.listdir(pics_path)
    for i, name in enumerate(tqdm(pics)):
        pic = cv2.imread(os.path.join(pics_path, name))
        # cv2.imwrite(os.path.join(save_path, name.split('_')[0]+'_'+ name.split('_')[1]+'.jpg'), pic)
        # cv2.imwrite(os.path.join(save_path, name.split('_')[0] + '.jpg'), pic)
        cv2.imwrite(os.path.join(save_path, name.split('_00.')[0] + '.jpg'), pic)

if __name__ == '__main__':

    # python -m pytorch_fid D:\w\total\SR\CeleA_SR_x4\hq D:\w\SR-TEST\HR\Test_CelebA-HQ-gt

    # <http://vis-www.cs.umass.edu/lfw/#download> lfw
    # eaely_fid()
    # change_name()
    main()

    # import cv2
    # path = 'D:\w\SR-TEST\LR\Test_CelebA-HQ_test'
    # save_path = 'D:\w\SR-TEST\LR\Test_CelebA-HQ_test-x4-128'
    # low_x4 = 'D:\w\SR-TEST\LR\Test_CelebA-HQ_test-x4-512'
    # os.makedirs(save_path, exist_ok=True)
    # pics = os.listdir(path)
    # for i, pic in enumerate(tqdm(pics)):
    #     img = cv2.imread(os.path.join(path, pic))
    #     img = cv2.resize(img, (128, 128))
    #     cv2.imwrite(os.path.join(save_path, pic), img)
    #
    # os.makedirs(low_x4, exist_ok=True)
    # pics = os.listdir(save_path)
    # for i, pic in enumerate(tqdm(pics)):
    #     img = cv2.imread(os.path.join(save_path, pic))
    #     img = cv2.resize(img, (512, 512), cv2.INTER_CUBIC)  # INTER_LINEAR
    #     cv2.imwrite(os.path.join(low_x4, pic), img)

    # path = 'D:\Study\Datasets\lfw-deepfunneled'
    # save_path = 'D:\Study\Datasets\lfw-first'
    # os.makedirs(save_path, exist_ok=True)
    # flies = os.listdir(path)
    # for i, flie in enumerate(tqdm(flies)):
    #     pics = os.listdir(os.path.join(path, flie))
    #     for i, pic in enumerate(pics):
    #         if i == 0:
    #             img = cv2.imread(os.path.join(path, flie, pic))
    #             cv2.imwrite(os.path.join(save_path, pic), img)
    #         else:
    #             pass
'''<https://github.com/bitzpy/Blind-Face-Restoration-Benchmark-Datasets-and-a-Baseline-Model>'''





