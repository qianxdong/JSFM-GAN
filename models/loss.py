import torch
from torchvision import models
from utils import utils
from torch import nn
from torchvision.transforms import transforms

from PIL import Image
import numpy as np
import cv2


IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

def tv_loss(x):
    """
    Total Variation Loss.
    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            ) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


# ==============================================================
# The triplet loss aims to make the distance between SR and GT smaller than SR and LR
class Triplet_loss(nn.Module):
    def __init__(self, margin=0.2):
        super(Triplet_loss, self).__init__()

        self.margin = margin  # 0.2, 0.5

    def forward(self, gt, sr, lr):
        loss = 0
        for gtf, srf, lrf in zip(gt, sr, lr):
            pos_dist = (gtf - srf).pow(2).sum(1)
            neg_dist = (srf - lrf).pow(2).sum(1)
            lossf = nn.ReLU()(pos_dist - neg_dist + self.margin)
            loss = loss + lossf.mean()
        return loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-8

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    <https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, ('Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class PCPFeat(torch.nn.Module):
    """
    Features used to calculate Perceptual Loss based on ResNet50 features.
    Input: (B, C, H, W), RGB, [0, 1]
    """
    def __init__(self, weight_path, model='vgg'):
        super(PCPFeat, self).__init__()
        if model == 'vgg':
            self.model = models.vgg19(pretrained=False)
            self.build_vgg_layers()
        elif model == 'resnet':
            self.model = models.resnet50(pretrained=False)
            self.build_resnet_layers()

        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def build_resnet_layers(self):
        self.layer1 = torch.nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool,
                    self.model.layer1
                    )
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.features = torch.nn.ModuleList(
                [self.layer1, self.layer2, self.layer3, self.layer4]
                )
    
    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        feature_layers = [0, 3, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x)
        std  = torch.Tensor([0.229, 0.224, 0.225]).to(x)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        
        features = []
        for m in self.features:
            x = m(x)
            features.append(x)
        return features 


class PCPLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self, opt, layer=5, model='vgg', ):
        super(PCPLoss, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf, w in zip(x_feats, y_feats, self.weights): 
            loss = loss + self.mse(xf, yf.detach()) * w
        return loss 


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        # self.mse = torch.nn.SmoothL1Loss()
        # self.mse = torch.nn.L1Loss()

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.mse(xf, yf.detach()) 
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'RAHinge':
            pass
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean() 
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss  = - prediction.mean()
            return loss

# ==============================================================
        elif self.gan_mode == 'RAHinge':   # RelativisticAverageHinge and expect one
            if for_discriminator:
                if target_is_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean()
            else:
                if target_is_real:
                    loss = nn.ReLU()(1 + prediction).mean()
                else:
                    loss = nn.ReLU()(1 - prediction).mean()
            return loss
# ===============================================================

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
            return loss

        return loss


class RegionStyleLoss(nn.Module):
    def __init__(self, reg_num=19, eps=1e-8):#19
        super().__init__()
        self.reg_num = reg_num
        self.eps = eps 
        self.mse = nn.MSELoss()

    def __masked_gram_matrix(self, x, m):
        b, c, h, w = x.shape
        m = m.view(b, -1, h*w)
        x = x.view(b, -1, h*w)
        total_elements = m.sum(2) + self.eps
        x = x * m
        G = torch.bmm(x, x.transpose(1, 2))
        return G / (c * total_elements.view(b, 1, 1)) 

    def __layer_gram_matrix(self, x, mask):
        b, c, h, w = x.shape
        all_gm = []
        for i in range(self.reg_num):
            sub_mask = mask[:, i].unsqueeze(1) 
            gram_matrix = self.__masked_gram_matrix(x, sub_mask)
            all_gm.append(gram_matrix)
        return torch.stack(all_gm, dim=1)

    def forward(self, x_feats, y_feats, mask):
        loss = 0
        for xf, yf in zip(x_feats[2:], y_feats[2:]):
            tmp_mask = torch.nn.functional.interpolate(mask, xf.shape[2:])
            xf_gm = self.__layer_gram_matrix(xf, tmp_mask)
            yf_gm = self.__layer_gram_matrix(yf, tmp_mask)
            tmp_loss = self.mse(xf_gm, yf_gm.detach())
            loss = loss + tmp_loss
        return loss


class ComponentStyleLoss(nn.Module):
    def __init__(self, reg_num=3, eps=1e-8):#19
        super().__init__()
        self.reg_num = reg_num
        self.eps = eps
        self.mse = nn.MSELoss()

    def __masked_gram_matrix(self, x, m):
        b, c, h, w = x.shape
        m = m.view(b, -1, h*w)
        x = x.view(b, -1, h*w)
        total_elements = m.sum(2) + self.eps
        x = x * m
        G = torch.bmm(x, x.transpose(1, 2))
        return G / (c * total_elements.view(b, 1, 1))

    def __layer_gram_matrix(self, x, mask):
        b, c, h, w = x.shape
        all_gm = []
        for i in range(self.reg_num):
            sub_mask = mask[:, i].unsqueeze(1)
            gram_matrix = self.__masked_gram_matrix(x, sub_mask)
            all_gm.append(gram_matrix)
        return torch.stack(all_gm, dim=1)

    def forward(self, x_feats, y_feats, mask):
        loss = 0
        for xf, yf in zip(x_feats[2:], y_feats[2:]):
            tmp_mask = torch.nn.functional.interpolate(mask, xf.shape[2:])
            xf_gm = self.__layer_gram_matrix(xf, tmp_mask)
            yf_gm = self.__layer_gram_matrix(yf, tmp_mask)
            tmp_loss = self.mse(xf_gm, yf_gm.detach())
            loss = loss + tmp_loss
        return loss


if __name__ == '__main__':

    print(IS_HIGH_VERSION)
    x = torch.rand(1, 3, 4, 4)
    y = torch.rand(1, 3, 4, 4)

    loss = FocalFrequencyLoss()(x, y)
    print(loss)
