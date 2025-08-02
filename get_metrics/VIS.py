#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：VIS.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/4/20 15:41 
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import cv2


# from get_metrics.eva import test
# # from eva import test
# from tqdm import tqdm
# import csv
# import os
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def get_csv(csv_path, clos):
    import pandas as pd
    import numpy as np
    datas = pd.read_csv(csv_path, usecols=clos)
    datas = np.array(datas)
    return datas

def Show_FID():
    xmajorLocator = MultipleLocator(1)  # 将x主刻度标签设置为20的倍数
    # xmajorFormatter = FormatStrFormatter('%5.1f')  # 设置x轴标签文本的格式
    xminorLocator = MultipleLocator(0.5)  # 将x轴次刻度标签设置为5的倍数

    ymajorLocator = MultipleLocator(50)  # 将y轴主刻度标签设置为0.5的倍数
    # ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
    yminorLocator = MultipleLocator(10)  # 将此y轴次刻度标签设置为0.1的倍数




    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    # 341.3627,79.2047,
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = [i for i in range(11)]

    # ori = [341.3627, 53.7941,  44.8503,  41.1225,  37.9236,  34.1055,  35.2761,  33.8968, 30.6951, 32.6095,
    #        31.0809, 30.3347, 28.24233, 28.0053, 24.8297, 25.6165, 25.8738, 23.68116,
    #        23.7926, 24.0632, 23.31134, 22.3681, 22.4862, 22.5112, 22.8715, 23.51983]
    #
    # ours = [341.3627, 361.1655, 361.9568, 360.9465, 368.1691, 370.4044, 367.7988, 367.1567, 367.6339,
    #         364.4321, 353.2948, 216.1023, 72.9226, 46.5109, 35.2739, 32.1237, 33.1011, 29.5934,
    #         28.5405, 27.57611, 26.3143, 30.0982, 26.5636, 25.5125, 25.0755, 24.4404]
    # ourswo = [341.3627,412.7213, 408.9578, 414.2364, 412.0197, 406.0834, 404.0227, 403.8874, 397.0965,
    #           385.9307, 380.7650, 364.4321, 280.4665, 248.0701, 69.4575, 41.0318, 38.9461, 36.1013, 37.3963, 31.9356, 31.098,
    #           31.9881, 31.9881, 27.9888, 29.5735, 29.1083]

    # ori = [341.3627, 30.3347, 28.24233, 28.0053, 24.8297, 25.6165, 25.8738, 23.68116,
    #      23.7926, 24.0632, 23.31134, 22.3681, 22.4862, 22.5112, 22.8715, 23.51983]
    #
    # ours   = [341.3627, 216.1023, 72.9226, 46.5109, 35.2739, 32.1237, 33.1011, 29.5934,
    #         28.5405, 27.57611, 26.3143, 30.0982, 26.5636, 25.5125, 25.0755,  24.4404]
    #
    # ourswo = [341.3627, 364.4321, 280.4665, 248.0701, 69.4575, 41.0318, 38.9461, 36.1013, 37.3963, 31.9356, 31.098, 31.9881, 31.9881, 27.9888, 29.5735, 29.1083]

    oursearly   = [341.3627, 361.1655, 361.9568, 360.9465, 368.1691, 370.4044, 367.7988, 367.1567,
                   # 367.6339, 364.4321, 363.2948, 359.8768, 354.5721, 348.6345, 341.8675, 293.8035]
                   367.6339, 364.4321, 353.2948]
    oursearlywo = [341.3627, 412.7213, 408.9578, 414.2364, 412.0197, 406.0834, 404.0227, 403.8874,
                   # 397.0965, 385.9307, 380.7650, 332.6200, 309.1963, 302.3983, 292.0996, 288.7951]
                   397.0965, 385.9307, 380.7650]

    oriearly   =  [341.3627, 53.7941,  44.8503,  41.1225,  37.9236,  34.1055,  35.2761,  33.8968,
                   # 30.6951,  32.6095,  31.0809,  31.0194,  31.6355,  30.7348,  27.2814,  28.6065]
                   30.6951, 32.6095, 31.0809]
    figure, ax = plt.subplots(figsize=(15, 10))
    # 设置主刻度标签的位置,标签文本的格式
    ax.xaxis.set_major_locator(xmajorLocator)
    # ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)

    # 显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    ax.xaxis.grid(True, which='major', linestyle='-.')  # x坐标轴的网格使用主刻度
    # ax.xaxis.grid(True, which='minor', linestyle='-.')  # x坐标轴的网格使用次刻度
    ax.yaxis.grid(True, which='major', linestyle='-.')  # y坐标轴的网格使用主刻度
    # ax.yaxis.grid(True, which='minor', linestyle='-.')  # y坐标轴的网格使用次刻度

    # plt.plot(x, ori, "r", linewidth=5.0, marker='*', ms=20, label="PSFRGAN")
    # plt.plot(x, oursearly, "b", linewidth=4.0, marker='.', ms=15, label="Ours")
    # plt.plot(x, ourswo, "g", linewidth=4.0, marker='.', ms=15, label="Ours-w/o")

    x = [i for i in range(11)]
    x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 100)
    ours_smooth = make_interp_spline(np.array(x), np.array(oursearly), k=1)(x_smooth)
    plt.plot(x_smooth, ours_smooth, "b", linewidth=3.0, marker='', ms=15, label="Ours")

    x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 100)
    ours_smooth = make_interp_spline(np.array(x), np.array(oursearlywo), k=1)(x_smooth)
    plt.plot(x_smooth, ours_smooth, "g", linewidth=3.0, marker='', ms=15, label="Ours-w/o")

    x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 100)
    ours_smooth = make_interp_spline(np.array(x), np.array(oriearly), k=1)(x_smooth)
    plt.plot(x_smooth, ours_smooth, "r", linewidth=5.0, marker='', ms=15, label="PSFRGAN")

    # plt.plot(x, ori, "r", linewidth=5.0, marker='*', ms=20, label="PSFRGAN")
    # plt.plot(x, ours, "b", linewidth=5.0, marker='.', ms=15, label="Ours")
    # x = [i for i in range(16)]

    # x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 100)
    # ori_smooth = make_interp_spline(x, np.array(ori), k=1)(x_smooth)
    # plt.plot(x_smooth, ori_smooth, "r", linewidth=5.0, marker='', ms=20, label="PSFRGAN")
    #
    # x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 100)
    # ours_smooth = make_interp_spline(np.array(x), np.array(ours), k=5)(x_smooth)
    # plt.plot(x_smooth, ours_smooth, "b", linewidth=3.0, marker='', ms=15, label="Ours")
    #
    # x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 25)
    # ours_smooth = make_interp_spline(np.array(x), np.array(ourswo), k=3)(x_smooth)
    # plt.plot(x_smooth, ours_smooth, "g", linewidth=3.0, marker='', ms=15, label="Ours-w/o")


    ##########################################################
    # 设置xtick和ytick的方向：in、out、inout
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.grid(linestyle='-.')
    plt.xticks(rotation=0)
    plt.tick_params(labelsize=30)
    plt.xlabel("Iterations(k)", font)
    plt.ylabel("FID", font)
    plt.title("FIDs in the training phase", fontdict=font)
    plt.legend(loc="lower right", prop=font)
    # plt.legend(loc="upper right", prop=font)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    # for x1, y1 in zip(x, ours):
    #     plt.text(x1, y1 + 1, str(round(y1,2)), ha='center', va='bottom', fontsize=15, rotation=30)
    plt.savefig("Show_FID.png")
    plt.show()

    # # coding: utf-8
    # import matplotlib.pyplot as plt
    #
    # # figsize = 11, 9
    # # figure, ax = plt.subplots(figsize = figsize)
    # x1 = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    # y1 = [0, 223, 488, 673, 870, 1027, 1193, 1407, 1609, 1791, 2113, 2388]
    # x2 = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    # y2 = [0, 214, 445, 627, 800, 956, 1090, 1281, 1489, 1625, 1896, 2151]
    #
    # # 设置输出的图片大小
    # figsize = 11, 9
    # figure, ax = plt.subplots(figsize=figsize)
    #
    # # 在同一幅图片上画两条折线
    # A, = plt.plot(x1, y1, '-r', label='A', linewidth=5.0)
    # B, = plt.plot(x2, y2, 'b-.', label='B', linewidth=5.0)
    #
    # # 设置图例并且设置图例的字体及大小
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 23,
    #          }
    # legend = plt.legend(handles=[A, B], prop=font1)
    #
    # # 设置坐标刻度值的大小以及刻度值的字体
    # plt.tick_params(labelsize=23)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # # print labels
    # [label.set_fontname('Times New Roman') for label in labels]
    # # 设置横纵坐标的名称以及对应字体格式
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 30,
    #          }
    # plt.xlabel('round', font2)
    # plt.ylabel('value', font2)
    # plt.show()

def showerr():
    from tqdm import tqdm
    for i in tqdm(range(2000)):
        gt = fr'D:\w\SR-TEST\HR\Test_CelebA-HQ-gt/{i}.jpg'
        lr = fr'D:\w\SR-TEST\LR\Test_CelebA-HQ_test/{i}.jpg'
        ps = fr'D:\w\SR-TEST\SR\GFPGAN_SR\aligned_CelebA_SR_GFPGAN\restored_imgs/{i}.jpg'
        my = f'D:\w\SR-TEST\SR\my_SR\CelebA_SR_my\hq/{i}.jpg'
        db = fr'D:\w\tmp_pic\download\CeleA_SR\hq/{i}.jpg'
        gt_img = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2GRAY)
        lr_img = cv2.cvtColor(cv2.imread(lr), cv2.COLOR_BGR2GRAY)
        ps_img = cv2.cvtColor(cv2.imread(ps), cv2.COLOR_BGR2GRAY)
        my_img = cv2.cvtColor(cv2.imread(my), cv2.COLOR_BGR2GRAY)
        db_img = cv2.cvtColor(cv2.imread(db), cv2.COLOR_BGR2GRAY)
        # gt_img,my_img = cv2.normalize(gt_img, my_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        d = abs(gt_img - ps_img)
        ps = cv2.applyColorMap((d/2).astype(np.uint8), cv2.COLORMAP_HOT)
        d = abs(gt_img - my_img)
        my = cv2.applyColorMap((d / 2).astype(np.uint8), cv2.COLORMAP_HOT)
        lq = abs(gt_img - lr_img)
        lq = cv2.applyColorMap((lq / 2).astype(np.uint8), cv2.COLORMAP_HOT)

        d = abs(gt_img - db_img)
        d = cv2.applyColorMap((d / 2).astype(np.uint8), cv2.COLORMAP_HOT)
        t = np.hstack((lq, ps, my, d))


        cv2.imwrite(f'D:\w\SR-TEST\CAM\err/{i}.jpg', t)



if __name__ == '__main__':
    showerr()
    # Show_FID()

    # name_test = 'PixelShuffle_22'  #
    # y_axis_data1 = []
    # y_axis_data2 = []
    # y_axis_data3 = []
    # y_axis_data4 = []
    # y_axis_data5 = []
    # x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # if os.path.exists(f'{name_test}.csv'):
    #     y_datas = get_csv(f'{name_test}).csv', clos=['SSIM', 'PSNR', 'LPIPS', 'NIQE'])
    #     for i in range(y_datas.shape[0]):
    #         y_axis_data1.append(y_datas[i][0])
    #         y_axis_data2.append(y_datas[i][1])
    #         y_axis_data3.append(y_datas[i][2])
    #         y_axis_data4.append(y_datas[i][3])
    # else:
    #     y_datas = []
    #     for i in tqdm(x_axis):
    #         gt = r'D:\Study\Codes\PSFR-GAN\img\test_out\gt'
    #         pic = f'D:/Study/Codes/PSFR-GAN/img/test_out/{name_test}/{i}/hq'
    #         result = test(pic, gt)
    #         y_datas.append(result)
    #         del result
    #
    #     for j in range(len(y_datas)):
    #         y_axis_data1.append(y_datas[j][0])
    #         y_axis_data2.append(y_datas[j][1])
    #         y_axis_data3.append(y_datas[j][2])
    #         y_axis_data4.append(y_datas[j][3])
    #         y_axis_data5.append(y_datas[j][4])
    #
    #     f = open(f'{name_test}.csv', 'w', newline='', encoding='utf-8')
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(["ID", "Iter(K)", "SSIM", "PSNR", "LPIPS", "NIQE", "cos_similarity"])  # 构建列表头
    #     for i in tqdm(range(len(y_axis_data1))):
    #         csv_writer.writerow([i, (i+1), y_axis_data1[i], y_axis_data2[i], y_axis_data3[i], y_axis_data4[i], y_axis_data5[i]])
    #     f.close()
    #
    #     # 画图 mssim, mpsnr, lpips, Niqe, cos_similarity
    #     # plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    #     # plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
    # plt.figure(figsize=(10, 8), dpi=200)
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(x_axis, y_axis_data1, 'yo--', alpha=0.5, linewidth=2, label='`|`ssim')
    # plt.xlabel('Iter(K)')
    # plt.ylabel('metrics')
    # # plt.title('`|`Ssim')
    # plt.legend()
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(x_axis, y_axis_data2, 'b*--', alpha=0.5, linewidth=2, label='`|`psnr')
    # plt.xlabel('Iter(K)')
    # plt.ylabel('metrics')
    # # plt.title('`|`Psnr')
    # plt.legend()
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(x_axis, y_axis_data3, 'rs--', alpha=0.5, linewidth=2, label='.|.lpips')
    # plt.xlabel('Iter(K)')
    # plt.ylabel('metrics')
    # # plt.title('.|.Lpips')
    # plt.legend()
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(x_axis, y_axis_data4, 'go--', alpha=0.5, linewidth=2, label='.|.Niqe')
    # plt.xlabel('Iter(K)')
    # plt.ylabel('metrics')
    # # plt.title('.|.Niqe')
    # plt.legend()
    #
    # plt.savefig(f"D:\Study\Codes\PSFR-GAN\get_metrics\{name_test}.png")
    # plt.show()


    # plt.hist(data,bins=40,facecolor='blue',edgecolor='red') 直方图
    # plt.bar(attr,v1,width=0.4, alpha=0.8, color='red', label="v1")
    # plt.pie(x=d,explode=explode,labels=attr,autopct = '%3.2f%%', colors=('b', 'g', 'r', 'c', 'm', 'y'))

    ## 设置数据标签位置及大小
    # for a, b in zip(x_axis_data, y_axis_data1):
    #     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
    # for a, b1 in zip(x_axis_data, y_axis_data2):
    #     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
    # for a, b2 in zip(x_axis_data, y_axis_data3):
    #     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
    # plt.legend()  # 显示上面的label
    # plt.scatter( x, y, color='b', label='Y=2*X^2' ) #主要是这里plot换成scatter，下面的根据需要修改
    # plt.xticks( x[::5], x_label[::5])
    # plt.yticks(z[::5])  #5是步长
    # plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度

    # plt.bar( x, y, width=0.2, color='r', label='Bar_1' ) #对比
    # #加一个柱状图，[i+0.2 for i in x]为间距生成式
    # plt.bar( [i+0.2 for i in x], z, width=0.2, color='b', label='Bar_2' )



