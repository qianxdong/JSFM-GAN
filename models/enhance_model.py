import os
# import numpy as np
import collections
# import random
import torch
import torch.nn as nn
import torch.optim as optim
# from adabelief_pytorch import AdaBelief

from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils


class EnhanceModel(BaseModel):
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--parse_net_weight', type=str, default='./pretrain_models/parse_multi_iter_90000.pth', help='parse model path')
            # parser.add_argument('--psfr_net_weight', type=str, default='./check_points/iter_300000_net_G.pth', help='remove model path')
            parser.add_argument('--lambda_pix', type=float, default=10.0, help='weight for parsing map')  # 10
            parser.add_argument('--lambda_pcp', type=float, default=1.0, help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_fm', type=float, default=10.0, help='weight for sr')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for sr')
            parser.add_argument('--lambda_ss', type=float, default=1000., help='weight for global style')

            parser.add_argument('--lambda_hf', type=float, default=1000., help='weight for HF loss')  # 1000
            parser.add_argument('--lambda_tri', type=float, default=1.0, help='weight for tri loss')  # 0
            parser.add_argument('--lambda_fpn', type=float, default=5.0, help='weight for fpn loss')  # 0
            parser.add_argument('--lambda_ffl', type=float, default=1000.0, help='weight for fft loss')  # 0
            parser.add_argument('--ref', type=int, default=3, help='ref chanle 3|6|22')
            parser.add_argument('--D_update_ratio', type=int, default=2, help='use for small batch')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netP = networks.define_P(opt, weight_path=opt.parse_net_weight)
        self.netG = networks.define_G(opt, use_norm='spectral_norm')
        # self.netG = networks.G_remove(opt, use_norm='spectral_norm')

        if self.isTrain:
            self.netD = networks.define_D(opt, opt.Dinput_nc, use_norm='spectral_norm')
            self.vgg_model = loss.PCPFeat(weight_path='./pretrain_models/vgg19-dcbb9e9d.pth').to(opt.device)
            if len(opt.gpu_ids) > 0:
                self.vgg_model = torch.nn.DataParallel(self.vgg_model, opt.gpu_ids, output_device=opt.device)

        self.model_names  = ['G']
        self.loss_names   = ['total', 'FPN', 'Pix', 'PCP', 'G', 'FM', 'D', 'SS', 'HF', 'FFL']  # Generator loss, fm loss, parsing loss, discriminator loss
        self.loss_G_names = ['total', 'FPN', 'Pix', 'PCP', 'G', 'FM', 'SS', 'HF', 'FFL']
        self.loss_D_names = ['total', 'FPN', 'Pix', 'PCP', 'G', 'FM', 'D',  'SS', 'HF', 'FFL']

        self.visual_names = ['img_LR', 'img_HR', 'img_SR', 'ref_Parse', 'hr_mask']
        self.fm_weights = [1**x for x in range(opt.D_num)]
        self.loss_class = 0

        if self.isTrain:
            self.model_names = ['G', 'D']
            self.load_model_names = ['G', 'D']
            # self.model_names = ['G', 'D', 'PD']
            # self.load_model_names = ['G', 'D', 'PD']

            self.criterionParse = torch.nn.CrossEntropyLoss().to(opt.device)
            self.criterionFFL = loss.FocalFrequencyLoss(loss_weight=1.0, alpha=1.0).to(opt.device)
            self.criterionFM = loss.FMLoss().to(opt.device)
            self.criterionHF = loss.HF_loss().to(opt.device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix = nn.L1Loss()
            # self.criterionTri = loss.Triplet_loss(margin=0.5).to(opt.device)
            self.criterionFft = loss.fft_loss().to(opt.device)
            self.criterionRS = loss.RegionStyleLoss()

            # self.optimizer_G = AdaBelief([p for p in self.netG.parameters() if p.requires_grad], lr=opt.g_lr, eps=1e-16, betas=(opt.beta1, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
            # self.optimizer_D = AdaBelief([p for p in self.netD.parameters() if p.requires_grad], lr=opt.d_lr, eps=1e-16, betas=(opt.beta1, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
            self.optimizer_G = optim.Adam([p for p in self.netG.parameters() if p.requires_grad], lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = optim.Adam([p for p in self.netD.parameters() if p.requires_grad], lr=opt.d_lr, betas=(opt.beta1, 0.999))
            # self.optimizer_PD = optim.Adam([p for p in self.netPD.parameters() if p.requires_grad], lr=opt.d_lr/10, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            # self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_PD]
            self.optimizers_G = [self.optimizer_G]
            self.optimizers_D = [self.optimizer_D]


    def eval(self):
        self.netG.eval()
        self.netP.eval()

    def load_pretrain_models(self,):
        self.netP.eval()
        print('Loading pretrained LQ face parsing network from', self.opt.parse_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netP.module.load_state_dict(torch.load(self.opt.parse_net_weight))
        else:
            self.netP.load_state_dict(torch.load(self.opt.parse_net_weight))
        self.netG.eval()
        print('Loading pretrained PSFRGAN from', self.opt.psfr_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netG.module.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)
        else:
            self.netG.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)
    
    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        if self.opt.ref == 25:
            self.img_LR = input['LR'].to(self.opt.device)
            self.img_HR = input['HR'].to(self.opt.device)
            self.hr_mask = input['Mask'].to(self.opt.device)
        elif self.opt.ref == 9:
            self.img_LR = input['LR'].to(self.opt.device)
            self.img_HR = input['HR'].to(self.opt.device)
            self.img_LR_Y = input['LR_Y'].to(self.opt.device)
            self.img_HR_Y = input['HR_Y'].to(self.opt.device)
            self.hr_mask = input['Mask'].to(self.opt.device)
        elif self.opt.ref == 6:
            self.img_LR = input['LR'].to(self.opt.device)
            self.img_HR = input['HR'].to(self.opt.device)
            self.hr_mask = input['Mask'].to(self.opt.device)
            # self.img_Ref = input['Ref'].to(self.opt.device)

        if self.opt.debug:
            print('SRNet input shape:', self.img_LR.shape, self.img_HR.shape)

    def forward(self):
        if self.opt.ref == 6:
            # with torch.no_grad():
            #     ref_mask, _ = self.netP(self.img_LR)
            #     self.ref_mask_onehot = (ref_mask == ref_mask.max(dim=1, keepdim=True)[0]).float().detach()
            if self.opt.debug:
                print('SRNet reference mask shape:', self.ref_mask_onehot.shape)
            # self.img_SR, self.SR_faets = self.netG(self.img_LR, self.img_HR, self.ref_mask_onehot, self.cur_iters)
            # self.img_SR = self.netG(self.img_LR, self.ref_mask_onehot)
            self.img_SR = self.netG(self.img_LR)

        elif self.opt.ref == 9:
            if self.opt.debug:
                print('SRNet reference no mask but Y', self.img_LR_Y.shape)
            # self.img_SR,self.SR_faets = self.netG(self.img_LR, self.img_LR_Y, None, self.cur_iters)
            self.img_SR, self.SR_faets = self.netG(self.img_LR, self.img_HR, self.img_LR_Y, self.cur_iters)
            self.img_SR_Y = (self.img_SR.detach()-torch.mean(self.img_SR.detach()))/torch.std(self.img_SR.detach())
            self.ref_mask_onehot = self.img_LR_Y
        else:
            if self.opt.debug:
                print('SRNet reference no mask')
            self.img_SR, self.SR_faets= self.netG(self.img_LR, self.img_Ref, None, self.cur_iters)
            self.ref_mask_onehot = torch.zeros(self.opt.batch_size, 19, 512, 512, device=self.opt.device)

        # with torch.no_grad():
        #     sr_mask, _ = self.netP(self.img_SR)
        #     self.sr_mask_onehot = (sr_mask == sr_mask.max(dim=1, keepdim=True)[0]).float().detach()

        if self.opt.gan_mode == 'RAHinge':
            if self.opt.ref == 22 and self.opt.Dinput_nc == 22:
                self.real_D_results_D = self.netD(torch.cat((self.img_HR, self.hr_mask), dim=1),return_feat=True)  # Dinput_nc=22
                self.fake_D_results_D = self.netD(torch.cat((self.img_SR.detach(), self.hr_mask), dim=1), return_feat=False)
                self.real_D_results   = self.netD(torch.cat((self.img_HR, self.hr_mask), dim=1), return_feat=True)
                self.fake_D_results   = self.netD(torch.cat((self.img_SR.detach(), self.hr_mask), dim=1), return_feat=False)
                self.fake_G_results   = self.netD(torch.cat((self.img_SR, self.hr_mask), dim=1), return_feat=True)
            elif self.opt.ref == 9 and self.opt.Dinput_nc == 6:
                self.real_D_results_D = self.netD(torch.cat((self.img_HR, self.img_HR_Y), dim=1),return_feat=True)  # Dinput_nc=6
                self.fake_D_results_D = self.netD(torch.cat((self.img_SR.detach(), self.img_HR_Y), dim=1), return_feat=False)
                self.real_D_results   = self.netD(torch.cat((self.img_HR, self.img_HR_Y), dim=1), return_feat=True)
                self.fake_D_results   = self.netD(torch.cat((self.img_SR.detach(), self.img_HR_Y), dim=1), return_feat=False)
                self.fake_G_results   = self.netD(torch.cat((self.img_SR, self.img_HR_Y), dim=1), return_feat=True)
            elif self.opt.ref == 9 and self.opt.Dinput_nc == 3:
                shape = self.img_HR.size()
                noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
                torch.randn(shape, out=noise)
                self.real_D_results_D = self.netD(self.img_HR + noise, return_feat=True)  # Dinput_nc=3
                self.fake_D_results_D = self.netD(self.img_SR.detach(), return_feat=False)
                self.real_D_results   = self.netD(self.img_HR, return_feat=True)
                self.fake_D_results   = self.netD(self.img_SR.detach(), return_feat=False)
                self.fake_G_results   = self.netD(self.img_SR, return_feat=True)
            else:
                raise ValueError('check opt.ref and opt.Dinput_nc !')
        else:
            if self.opt.Dinput_nc == 22:
                self.real_D_results = self.netD(torch.cat((self.img_HR, self.hr_mask), dim=1), return_feat=True)  # Dinput_nc=22
                self.fake_D_results = self.netD(torch.cat((self.img_SR.detach(), self.hr_mask), dim=1), return_feat=False)
                self.fake_G_results = self.netD(torch.cat((self.img_SR, self.hr_mask), dim=1), return_feat=True)
            elif self.opt.Dinput_nc == 83:   # vgg19 encode
                self.real_D_results = self.netD(torch.cat((self.img_HR_feats[0], self.hr_mask), dim=1), return_feat=True)  # Dinput_nc=22
                self.fake_D_results = self.netD(torch.cat((self.img_SR_feats[0].detach(), self.hr_mask), dim=1), return_feat=False)
                self.fake_G_results = self.netD(torch.cat((self.img_SR_feats[0], self.hr_mask), dim=1), return_feat=True)
            elif self.opt.ref == 9 and self.opt.Dinput_nc == 6:
                # self.D_noises = self.D_noise.repeat(self.opt.batch_size, 1, 1, 1)
                self.real_D_results = self.netD(torch.cat((self.img_HR, self.img_HR_Y), dim=1), return_feat=True)
                self.fake_D_results = self.netD(torch.cat((self.img_SR.detach(), self.img_HR_Y), dim=1), return_feat=False)  # 不返回feat
                self.fake_G_results = self.netD(torch.cat((self.img_SR, self.img_HR_Y), dim=1), return_feat=True)
            elif self.opt.ref == 6 and self.opt.Dinput_nc == 3:
                self.real_D_results = self.netD(self.img_HR, return_feat=True)
                self.fake_D_results = self.netD(self.img_SR.detach(), return_feat=False)
                self.fake_G_results = self.netD(self.img_SR, return_feat=True)
            else:
                raise ValueError('check opt.ref and opt.Dinput_nc !')

        self.img_SR_feats = self.vgg_model(self.img_SR)
        self.img_HR_feats = self.vgg_model(self.img_HR)

    def backward_G(self):
        # tmp_loss = 0
        # for i, feat in enumerate(self.SR_faets):
        #     if feat.shape[-1] != self.img_HR.shape[-1]:
        #         img_HR = nn.functional.interpolate(self.img_HR, feat.shape[2:], mode='bicubic', align_corners=False)
        #     else:
        #         img_HR = self.img_HR
        #     loss_f = self.criterionPix(feat, img_HR)
        #     tmp_loss = loss_f + tmp_loss
        # self.loss_FPN = tmp_loss * self.opt.lambda_fpn
        self.loss_FPN = 0

        # Triplet Loss
        # self.loss_Tri = self.criterionTri(self.img_HR, self.img_SR, self.img_LR)*self.opt.lambda_tri

        self.loss_FFL = 0
        # self.loss_FFL = self.criterionFFL(self.img_SR, self.img_HR)*self.opt.lambda_ffl

        # Pix Loss
        # self.loss_Pix = self.criterionPix(self.img_SR, self.img_HR) * (self.opt.lambda_pix*(torch.log(self.criterionPix(self.img_LR, self.img_HR)+1) + 1).item())
        self.loss_Pix = self.criterionPix(self.img_SR, self.img_HR)* self.opt.lambda_pix

        # semantic style loss
        self.loss_SS = self.criterionRS(self.img_SR_feats, self.img_HR_feats, self.hr_mask) * self.opt.lambda_ss

        # perceptual loss
        self.loss_PCP = self.criterionPCP(self.img_SR_feats, self.img_HR_feats) * self.opt.lambda_pcp

        # high frequency loss
        # self.loss_HF = self.criterionHF(self.img_SR, self.img_HR) * self.opt.lambda_hf
        self.loss_HF = 0
        # self.loss_Fft = self.criterionFft(self.img_SR, self.img_HR)*self.opt.lambda_fft
        # self.loss_Fft = 0.

        # Feature matching loss
        tmp_loss = 0
        for i, w in zip(range(self.opt.D_num), self.fm_weights):
            tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) * w
        self.loss_FM = tmp_loss * self.opt.lambda_fm / self.opt.D_num

        # Generator loss
        tmp_loss = 0
        for i in range(self.opt.D_num):
            if self.opt.gan_mode == 'RAHinge':
                tmp_loss = tmp_loss + \
                           0.5 * (self.criterionGAN(self.real_D_results[i][0].detach() - torch.mean(self.fake_G_results[i][0].detach()), True, for_discriminator=False) + \
                                  self.criterionGAN(self.fake_G_results[i][0] - torch.mean(self.real_D_results[i][0]), False, for_discriminator=False))
            else:
                tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
        self.loss_G = tmp_loss * self.opt.lambda_g / self.opt.D_num

        # self.loss_PG = 0
        # for i in range(self.opt.PD_num):
        #     self.loss_PG = self.loss_PG = 0 + self.criterionGAN(self.fake_PG_results[i][0], True, for_discriminator=False)
        # self.loss_PG = self.loss_PG/self.opt.PD_num

        # self.loss_PG = 0
        # for i in range(self.opt.PD_num):
        #     for j in range(len(self.fake_PG_results)):
        #         self.loss_PG = self.loss_PG + self.criterionGAN(self.fake_PG_results[j][i][0], True, for_discriminator=False)
        # self.loss_PG = self.loss_PG / self.opt.PD_num / 4

        self.loss_total = self.loss_Pix + self.loss_PCP + self.loss_FM + self.loss_G + self.loss_SS + self.loss_HF + self.loss_FPN + self.loss_FFL
        self.loss_total.backward()

    def backward_D(self):
        self.loss_D = 0
        self.loss_tmp = 0
        for i in range(self.opt.D_num):
            if self.opt.gan_mode == 'RAHinge':
                self.loss_tmp += 0.5 * (self.criterionGAN(self.fake_D_results_D[i].detach()-torch.mean(self.real_D_results_D[i][0].detach()), False) +
                                  self.criterionGAN(self.real_D_results_D[i][0]-torch.mean(self.fake_D_results_D[i]), True))
            else:
                self.loss_D += 0.5 * (self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0], True))

        if self.opt.gan_mode == 'RAHinge':
            self.loss_D = self.loss_tmp / self.opt.D_num
            self.loss_D.backward()
        else:
            self.loss_D = self.loss_D / self.opt.D_num
            self.loss_D.backward()

    # def backward_PD(self):
    #     self.loss_PD = 0
    #     for i in range(self.opt.PD_num):
    #         for j in range(len(self.fake_PG_results)):
    #             self.loss_PD += 0.5 * (self.criterionGAN(self.fake_PD_results[j][i], False) +
    #                                    self.criterionGAN(self.real_PD_results[j][i][0], True))
    #     self.loss_PD = self.loss_PD / self.opt.PD_num / 4
    #     self.loss_PD.backward()

    def optimize_parameters(self):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D ------------
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # # ---- Update D ------------
        # self.optimizer_PD.zero_grad()
        # self.backward_PD()
        # self.optimizer_PD.step()

    def get_current_visuals(self, size=512):
        out = []
        visual_imgs = []

        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        # if self.opt.ref == 6:
            # out.append(utils.tensor_to_numpy(self.img_LR_Y))
            # out.append(utils.tensor_to_numpy(self.img_SR_Y))
            # out.append(utils.tensor_to_numpy(self.img_HR_Y))
            # out.append(utils.tensor_to_numpy(self.img_Ref))
        # HRh, _ = utils.HF_HV(self.img_HR)
        # SRh, _ = utils.HF_HV(self.img_SR)
        #
        # if len(HRh) == 1:
        #     out.append(utils.tensor_to_numpy(SRh[0]))
        #     out.append(utils.tensor_to_numpy(HRh[0]))
        # else:
        # #
        # # for i in range(len(HRh)):
        # #     print(utils.tensor_to_numpy(SRh[i]).shape)
        # #     out.append(utils.tensor_to_numpy(SRh[i]))
        # #     out.append(utils.tensor_to_numpy(HRh[i]))
        # torch.cat([a1, a2], dim=0

        out_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        visual_imgs += out_imgs
        if self.opt.ref == 6:
            visual_imgs.append(utils.color_parse_map(self.ref_mask_onehot, size))

        # visual_imgs.append(utils.color_parse_map(self.sr_mask_onehot, size))

        if self.opt.ref == 6:
            visual_imgs.append(utils.color_parse_map(self.hr_mask, size))
        return visual_imgs

    def get_gragh(self,):
        visual_gragh = []
        G = self.netG
        G_INPUT = (self.img_LR, self.ref_mask_onehot)
        visual_gragh.append(G)
        visual_gragh.append(G_INPUT)
        return visual_gragh

    def optimize_parameters_D(self):
        #  Update G and Update D based on the loss of D
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # if self.loss_FM < 4.:
        #     self.optimizer_D.zero_grad()
        #     self.backward_D()
        #     self.optimizer_D.step()
        # else:
        #     self.backward_D()


        # self.optimizer_PD.zero_grad()
        # self.backward_PD()
        # self.optimizer_PD.step()


    def optimize_parameters_G(self):
        #  Update G and Gradient of D accumulation
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.backward_D()


