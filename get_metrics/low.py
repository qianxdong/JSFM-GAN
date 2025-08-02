
import numpy as np
def get_class_pix_percent(pic_rgb):
    MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255],
                     [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0],
                     [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                     [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow',
                  'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip',
                  'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    im = np.array(pic_rgb)
    h, w, d = im.shape
    r, g, b = im[..., 0], im[..., 1], im[..., 2]  # 分成红绿蓝三个通道
    pixs = []
    percent = []
    for i in range(len(MASK_COLORMAP)-1):
        clas = im[(r == MASK_COLORMAP[i+1][0]) & (g == MASK_COLORMAP[i+1][1]) & (b == MASK_COLORMAP[i+1][2])]
        pixs.append(clas.shape[0])
        percent.append(round(100 * clas.shape[0] / (w * h), 4))
        # print(f'{label_list[i]}区域像素数：{clas.shape[0]}，占比{round(100 * clas.shape[0] / (w * h),4)}')
    return pixs, percent, label_list






