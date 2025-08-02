import os
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from data.image_folder import make_dataset

import torch
from torchvision.transforms import transforms
from data.base_dataset import BaseDataset
from utils.utils import onehot_parse_map


class FFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.Pimg_size
        self.lr_size = opt.Gin_size
        self.hr_size = opt.Gout_size
        self.shuffle = True if opt.isTrain else False 

        self.img_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'imgs512')))
        self.mask_dataset = sorted(make_dataset(os.path.join(opt.dataroot, 'masks512')))

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.to_tensor_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.random_crop = transforms.RandomCrop(self.hr_size)

    def __len__(self,):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img_path = self.img_dataset[idx]
        mask_path = self.mask_dataset[idx]

        hr_img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')

        hr_img = hr_img.resize((self.hr_size, self.hr_size))
        hr_img = self.random_gray(hr_img, p=0.3)
        scale_size = np.random.randint(32, 256)
        lr_img = self.complex_imgaug(hr_img, self.img_size, scale_size)
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        mask_img = mask_img.resize((self.hr_size, self.hr_size))
        mask_label = onehot_parse_map(mask_img)
        mask_label = torch.tensor(mask_label).float()
        return {'HR': hr_tensor, 'LR': lr_tensor, 'Mask': mask_label}

    @staticmethod
    def complex_imgaug(x, org_size, scale_size):
        """input single RGB PIL Image instance"""
        x = np.array(x)
        x = x[np.newaxis, :, :, :]
        aug_seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.GaussianBlur((3, 15)),
                    iaa.AverageBlur(k=(3, 15)),
                    iaa.MedianBlur(k=(3, 15)),
                    iaa.MotionBlur((5, 25))
                ])),
                iaa.Resize(scale_size, interpolation=ia.ALL),
                iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)),
                iaa.Sometimes(0.7, iaa.JpegCompression(compression=(10, 65))),
                iaa.Resize(org_size),
            ])

        aug_img = aug_seq(images=x)
        return aug_img[0]

    @staticmethod
    def random_gray(x, p=0.3):
        """input single RGB PIL Image instance"""
        x = np.array(x)
        x = x[np.newaxis, :, :, :]
        aug = iaa.Sometimes(p, iaa.Grayscale(alpha=1.0))
        aug_img = aug(images=x)
        return aug_img[0]

    @staticmethod
    def adjust_mask(mask_label):
        # part of mask
        skin = mask_label[1:2, :, :]
        eye_g = mask_label[3:4, :, :]
        l_eye = mask_label[4:5, :, :]
        r_eye = mask_label[5:6, :, :]
        l_brow = mask_label[6:7, :, :]
        r_brow = mask_label[7:8, :, :]
        
        # to numpy
        skin = torch.from_numpy(skin).float()
        eye_g = torch.from_numpy(eye_g).float()
        l_eye = torch.from_numpy(l_eye).float()
        r_eye = torch.from_numpy(r_eye).float()
        l_brow = torch.from_numpy(l_brow).float()
        r_brow = torch.from_numpy(r_brow).float()

        c = torch.chunk(eye_g, chunks=2, dim=2)
        area = torch.sum(eye_g)
        area1 = torch.sum(c[0])
        area2 = torch.sum(c[1])

        # glass
        if area / (mask_label.shape[1] * mask_label.shape[2]) < 0.04 or area1 == 0 or area2 == 0:
            skin = skin + eye_g
            eye_g[eye_g > 0] = 0
            mask_label[1:2, :, :] = skin
            mask_label[3:4, :, :] = eye_g
        # eye    
        area_le = torch.sum(l_eye)
        area_re = torch.sum(r_eye)

        if (area_le+area_re) / (mask_label.shape[1] * mask_label.shape[2]) < 0.001 or area_le == 0 or area_re == 0:
            skin = skin + l_eye + r_eye
            l_eye[l_eye > 0] = 0
            r_eye[r_eye > 0] = 0
            mask_label[1:2, :, :] = skin
            mask_label[4:6, :, :] = torch.cat([l_eye, r_eye], dim=0)
            
        # brow    
        area_lb = torch.sum(l_brow)
        area_rb = torch.sum(r_brow)

        if (area_lb+area_rb) / (mask_label.shape[1] * mask_label.shape[2]) < 0.005 or area_lb == 0 or area_rb == 0:
            skin = skin + l_brow + r_brow
            l_brow[l_brow > 0] = 0
            r_brow[r_brow > 0] = 0
            mask_label[1:2, :, :] = skin
            mask_label[6:8, :, :] = torch.cat([l_brow, r_brow], dim=0)
        
        return mask_label
