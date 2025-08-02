#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：jsfmgan_model.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/7/5 20:43 
'''


import torch
import torch.nn as nn
import torch.optim as optim
# from adabelief_pytorch import AdaBelief

from models import loss
from models import networks
from .base_model import BaseModel
from utils import utils


class JSFMGANModel(BaseModel):
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--parse_net_weight', type=str, default='./pretrain_models/parse_multi_iter_90000.pth', help='parse model path')
            parser.add_argument('--lambda_pix', type=float, default=10.0, help='weight for pixel loss')  # 10
            parser.add_argument('--lambda_pcp', type=float, default=1.0,  help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_fm',  type=float, default=10.0, help='weight for Multi scale feature matching loss')
            parser.add_argument('--lambda_g',   type=float, default=1.0,  help='weight for adversarial loss')
            parser.add_argument('--lambda_ss',  type=float, default=1000., help='weight for semantic style loss')
            parser.add_argument('--lambda_srl', type=float, default=10.0,  help='weight for Stage reconstruction loss')
            parser.add_argument('--srl', action='store_true', help='if specified, set to fpn')
            parser.add_argument('--afpm', action='store_true', help='Adjustment of Facial Parsing Maps')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.netP = networks.define_P(opt, weight_path=opt.parse_net_weight)
        self.netG = networks.define_G(opt, use_norm='spectral_norm')
        if self.isTrain:
            self.netD = networks.define_D(opt, opt.Dinput_nc, use_norm='spectral_norm')
            self.vgg_model = loss.PCPFeat(weight_path='./pretrain_models/vgg19-dcbb9e9d.pth').to(opt.device)
            if len(opt.gpu_ids) > 0:
                self.vgg_model = torch.nn.DataParallel(self.vgg_model, opt.gpu_ids, output_device=opt.device)

        self.model_names  = ['G']
        self.loss_names   = ['total', 'SRL', 'Pix', 'PCP', 'G', 'FM', 'D', 'SS']
        self.loss_G_names = ['total', 'SRL', 'Pix', 'PCP', 'G', 'FM',      'SS']
        self.loss_D_names = ['total', 'SRL', 'Pix', 'PCP', 'G', 'FM', 'D', 'SS']
        self.visual_names = ['img_LR', 'img_SR','img_HR']
        self.fm_weights = [1 ** x for x in range(opt.D_num)]
        self.sf_weights = [1, 2, 4]

        if self.isTrain:
            self.model_names = ['G', 'D']
            self.load_model_names = ['G', 'D']
            self.criterionFM = loss.FMLoss().to(opt.device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix = nn.L1Loss().to(opt.device)
            self.criterionSRL = loss.L1_Charbonnier_loss().to(opt.device)
            self.criterionRS = loss.RegionStyleLoss().to(opt.device)

            self.optimizer_G = optim.Adam([p for p in self.netG.parameters() if p.requires_grad], lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = optim.Adam([p for p in self.netD.parameters() if p.requires_grad], lr=opt.d_lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def eval(self):
        self.netG.eval()
        self.netP.eval()

    def load_pretrain_models(self, ):
        self.netP.eval()
        print('Loading pretrained LQ face parsing network from', self.opt.parse_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netP.module.load_state_dict(torch.load(self.opt.parse_net_weight))
        else:
            self.netP.load_state_dict(torch.load(self.opt.parse_net_weight))

        self.netG.eval()
        print('Loading pretrained JSFMGAN from', self.opt.jsfm_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netG.module.load_state_dict(torch.load(self.opt.jsfm_net_weight), strict=False)
        else:
            self.netG.load_state_dict(torch.load(self.opt.jsfm_net_weight), strict=False)

    def set_input(self, input):
        self.img_LR = input['LR'].to(self.opt.device)
        self.img_HR = input['HR'].to(self.opt.device)
        self.hr_mask = input['Mask'].to(self.opt.device)

        if self.opt.debug:
            print('SRNet input shape:', self.img_LR.shape, self.img_HR.shape)

    # afpm实现
    def adjust_map(self, maps):
        # part of mask
        skin = maps[1:2, :, :]
        eye_g = maps[3:4, :, :]
        l_eye = maps[4:5, :, :]
        r_eye = maps[5:6, :, :]
        l_brow = maps[6:7, :, :]
        r_brow = maps[7:8, :, :]
        
        c = torch.chunk(eye_g, chunks=2, dim=2)
        area = torch.sum(eye_g)
        area1 = torch.sum(c[0])
        area2 = torch.sum(c[1])

        if area / (maps.shape[1] * maps.shape[2]) < 0.04 or area1 == 0 or area2 == 0:
            skin = skin + eye_g
            eye_g[eye_g > 0] = 0
            maps[1:2, :, :] = skin
            maps[3:4, :, :] = eye_g
        # eye    
        area_le = torch.sum(l_eye)
        area_re = torch.sum(r_eye)

        if (area_le+area_re) / (maps.shape[1] * maps.shape[2]) < 0.001 or area_le == 0 or area_re == 0:
            skin = skin + l_eye + r_eye
            l_eye[l_eye > 0] = 0
            r_eye[r_eye > 0] = 0
            maps[1:2, :, :] = skin
            maps[4:6, :, :] = torch.cat([l_eye, r_eye], dim=0)
            
        # brow    
        area_lb = torch.sum(l_brow)
        area_rb = torch.sum(r_brow)

        if (area_lb+area_rb) / (maps.shape[1] * maps.shape[2]) < 0.005 or area_lb == 0 or area_rb == 0:
            skin = skin + l_brow + r_brow
            l_brow[l_brow > 0] = 0
            r_brow[r_brow > 0] = 0
            maps[1:2, :, :] = skin
            maps[6:8, :, :] = torch.cat([l_brow, r_brow], dim=0)
        return maps

    def forward(self):
        # 获取人脸解析图
        with torch.no_grad():
            ref_mask, _ = self.netP(self.img_LR)
            self.ref_mask_onehot = (ref_mask == ref_mask.max(dim=1, keepdim=True)[0]).float().detach()
            # 对称调整人脸解析图
            if self.opt.afpm:
                self.ref_mask_onehot = self.adjust_map(self.ref_mask_onehot.to(self.opt.device))
            if self.opt.debug:
                print('SRNet reference mask shape:', self.ref_mask_onehot.shape)

        if self.opt.srl:
            # 保存阶段特征以计算阶段重建损失
            self.img_SR, self.SR_faets = self.netG(self.img_LR, self.ref_mask_onehot)
        else:
            self.img_SR = self.netG(self.img_LR, self.ref_mask_onehot)


        # 保存判别器的结果以计算对抗损失
        self.real_D_results = self.netD(torch.cat((self.img_HR, self.hr_mask), dim=1), return_feat=True)
        self.fake_D_results = self.netD(torch.cat((self.img_SR.detach(), self.hr_mask), dim=1), return_feat=False)
        self.fake_G_results = self.netD(torch.cat((self.img_SR, self.hr_mask), dim=1), return_feat=True)

        # 保存vgg的特征以计算感知损失
        self.img_SR_feats = self.vgg_model(self.img_SR)
        self.img_HR_feats = self.vgg_model(self.img_HR)

    def backward_G(self):
        # stage reconstruction loss
        if self.opt.srl:
            tmp_loss = 0
            for i, feat in enumerate(self.SR_faets):
                # 插值保证特征大小一致
                if feat.shape[-1] != self.img_HR.shape[-1]:
                    img_HR = nn.functional.interpolate(self.img_HR, feat.shape[2:],
                                                       mode='bicubic', align_corners=False)
                else:
                    img_HR = self.img_HR
                # 对不同尺寸阶段特征赋予不同的权重进行损失计算
                loss_f = self.criterionSRL(feat, img_HR)*self.sf_weights[i]
                tmp_loss = loss_f + tmp_loss
            self.loss_SRL = tmp_loss * self.opt.lambda_srl/len(self.SR_faets)
        else:
            self.loss_FPN = 0

        # Pix Loss
        self.loss_Pix = self.criterionPix(self.img_SR, self.img_HR)

        # semantic style loss
        self.loss_SS = self.criterionRS(self.img_SR_feats, self.img_HR_feats, self.hr_mask) * self.opt.lambda_ss

        # perceptual loss
        self.loss_PCP = self.criterionPCP(self.img_SR_feats, self.img_HR_feats) * self.opt.lambda_pcp

        # Feature matching loss
        tmp_loss = 0
        for i, w in zip(range(self.opt.D_num), self.fm_weights):
            tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) * w
        self.loss_FM = tmp_loss * self.opt.lambda_fm / self.opt.D_num

        # Generator loss
        tmp_loss = 0
        for i in range(self.opt.D_num):
            tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
        self.loss_G = tmp_loss * self.opt.lambda_g / self.opt.D_num

        self.loss_total = self.loss_Pix + self.loss_PCP + self.loss_FM + self.loss_G + \
                        + self.loss_SRL + self.loss_SS
        self.loss_total.backward()

    def backward_D(self):
        self.loss_D = 0
        self.loss_tmp = 0
        for i in range(self.opt.D_num):
            self.loss_D += 0.5 * (self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0], True))
        self.loss_D = self.loss_D / self.opt.D_num
        self.loss_D.backward()

    def optimize_parameters(self):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D ------------
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_visuals(self, size=512):
        out = []
        visual_imgs = []

        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        if self.opt.srl:
            out.append(utils.tensor_to_numpy(self.SR_faets[0]))
            out.append(utils.tensor_to_numpy(self.SR_faets[1]))
            out.append(utils.tensor_to_numpy(self.SR_faets[2]))


        out_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        visual_imgs += out_imgs
        visual_imgs.append(utils.color_parse_map(self.ref_mask_onehot, size))
        visual_imgs.append(utils.color_parse_map(self.hr_mask, size))
        return visual_imgs



