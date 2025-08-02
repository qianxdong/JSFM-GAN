from models.blocks import *


# 小尺寸的风格转化
class STNorm(nn.Module):
    def __init__(self, norm_nc, ref_nc, norm_type='spade', ksz=3):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.group_norm = nn.GroupNorm(2, norm_nc, affine=False)
        mid_c = 64
        self.norm_type = norm_type
        if norm_type == 'spade':
            self.conv1 = nn.Sequential(
                nn.Conv2d(ref_nc, mid_c, ksz, 1, ksz // 2),
                nn.LeakyReLU(0.2, True),
            )
            self.gamma_conv = nn.Conv2d(mid_c, norm_nc, kernel_size=(ksz, ksz), stride=1, padding=ksz // 2)
            self.beta_conv  = nn.Conv2d(mid_c, norm_nc, kernel_size=(ksz, ksz), stride=1, padding=ksz // 2)

    def get_gamma_beta(self, x, conv, gamma_conv, beta_conv):
        act = conv(x)
        gamma = gamma_conv(act)
        beta = beta_conv(act)
        return gamma, beta

    def forward(self, x, ref):
        normalized_input = self.param_free_norm(x)
        if x.shape[-1] != ref.shape[-1]:
            ref = nn.functional.interpolate(ref, x.shape[2:], mode='bicubic', align_corners=False)  # 双三次插值下采样
        if self.norm_type == 'spade':
            gamma, beta = self.get_gamma_beta(ref, self.conv1, self.gamma_conv, self.beta_conv)
            return normalized_input * gamma + beta
        elif self.norm_type == 'in':
            return normalized_input


# 非对称调制模块AMM
class AMM(nn.Module):
    def __init__(self, in_channel, out_channel, ref_nc, norm_type, **kwargs):
        super(AMM, self).__init__()
        self.input_c = in_channel
        self.output_c = out_channel
        self.conv1_3 = nn.Conv2d(self.input_c, self.output_c, kernel_size=(1, 3), padding=(0, 1), groups=1, bias=False)
        self.conv3_1 = nn.Conv2d(self.input_c, self.output_c, kernel_size=(3, 1), padding=(1, 0), groups=1, bias=False)
        self.conv3_3 = nn.Conv2d(self.input_c, self.output_c, kernel_size=(3, 3), padding=(1, 1), groups=1, bias=False)
        self.norm0 = JSFT(in_channel, ref_nc, norm_type)
        self.norm1 = JSFT(in_channel, ref_nc, norm_type)
        self.norm2 = JSFT(in_channel, ref_nc, norm_type)

    def forward(self, x, ref):
        x1 = self.norm0(self.conv1_3(x), ref)
        x2 = self.norm1(self.conv3_1(x), ref)
        x3 = self.norm2(self.conv3_3(x), ref)
        out = x1 + x2 + x3
        return out


# 上采样特征补充补充模块
class USFM(nn.Module):  #
    def __init__(self, in_c, ratio, ref_nc, norm_type, relu_type):
        super(USFM, self).__init__()

        self.input_c = in_c
        self.ratio = ratio
        self.output_c = self.input_c * self.ratio // 2
        self.conv1_1 = nn.Conv2d(self.input_c, self.output_c, kernel_size=(1, 1), padding=0, groups=1, bias=True)
        self.conv3_3 = nn.Conv2d(self.output_c, self.output_c, kernel_size=(3, 3), padding=1, groups=1, bias=True)
        self.amm = AMM(self.input_c, self.output_c, ref_nc, norm_type)
        self.norm = JSFT(in_c, ref_nc, norm_type)
        self.relu = ReluLayer(self.output_c, relu_type)

    def forward(self, x, ref):
        x1 = self.relu(self.norm(self.conv3_3(self.conv1_1(x)), ref))
        x2 = self.relu(self.amm(x, ref))
        out = torch.cat([x1, x2], dim=1)
        return out


# 添加噪声
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


# 上采样特征补充残差模块
class USFMResBlock(nn.Module):
    def __init__(self, cin, rate, ref_nc, uptype, norm_type='spade', relu_type='relu'):
        super(USFMResBlock, self).__init__()
        self.relu = ReluLayer(cin*rate, relu_type)
        self.cin = cin
        self.rate = rate
        self.uptype = uptype
        # 添加噪声
        self.noise = NoiseInjection()

        if self.uptype == 'usfm':
            # 亚像素卷积
            self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
            # 特征补充模块USFM
            self.usfm = USFM(self.cin, self.rate, ref_nc=ref_nc, norm_type=norm_type, relu_type=relu_type)  # CX2
        elif self.uptype == 'pixelshuffle':
            self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        else:
            self.amm = AMM(self.cin, self.cin*self.rate, ref_nc=ref_nc, norm_type=norm_type)  # C*rate

    def forward(self, x, ref):
        if self.uptype == 'usfm':
            x = self.relu(self.usfm(x, ref))
            out = self.pixelshuffle(x)
        elif self.uptype == 'pixelshuffle':
            out = self.pixelshuffle(x)
        elif self.uptype == 'bilinear':
            x = self.relu(self.amm(x, ref))
            out = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        elif self.uptype == 'nearest':
            x = self.amm(x, ref)
            out = nn.Upsample(scale_factor=2)(x)
        else:
            raise ValueError('Please choose the correct uptype from nearest | bilinear| pixelshuffle | usfm')
        out = self.noise(out, None)
        return out


# 联合阶段特征残差块
class JSFTResBlock(nn.Module):
    def __init__(self, fin, fout, ref_nc, relu_type, norm_type='spade'):
        super().__init__()
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=(3, 3), padding=1)
        self.conv_3 = nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=1)

        self.norm_0 = JSFT(fmiddle, ref_nc, norm_type)
        self.norm_3 = JSFT(fin, ref_nc, norm_type)

        self.relu   = ReluLayer(fmiddle, relu_type)

    def forward(self, x, ref):
        res = self.conv_0(self.relu(self.norm_0(x, ref)))
        res = self.conv_3(self.relu(self.norm_3(res, ref)))
        out = x + res
        return out


# 小尺寸的风格调制
class STResBlock(nn.Module):
    def __init__(self, fin, fout, ref_nc, relu_type, norm_type='spade'):
        super().__init__()
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        # define normalization layers
        self.norm_0 = STNorm(fmiddle, ref_nc, norm_type)
        self.norm_1 = STNorm(fmiddle, ref_nc, norm_type)
        self.relu   = ReluLayer(fmiddle, relu_type)

    def forward(self, x, ref):
        res = self.conv_0(self.relu(self.norm_0(x, ref)))
        res = self.conv_1(self.relu(self.norm_1(res, ref)))
        out = x + res
        return out


# 联合阶段特征转换
class JSFT(nn.Module):
    def __init__(self, norm_nc, ref_nc, norm_type='spade', ksz=3):
        super().__init__()
        ref_nc -= 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        mid_c = 64

        self.norm_type = norm_type
        if norm_type == 'spade':
            self.conv1 = nn.Sequential(
                nn.Conv2d(ref_nc, mid_c, ksz, 1, ksz // 2),
                nn.LeakyReLU(0.2, True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(ref_nc, mid_c, ksz, 1, ksz // 2),
                nn.LeakyReLU(0.2, True),
            )
            self.gamma_DI = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz // 2)
            self.beta_DI  = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz // 2)
            self.gamma_SF = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz // 2)
            self.beta_SF  = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz // 2)

    def get_gamma_beta(self, x, conv, gamma_conv, beta_conv):
        act = conv(x)
        gamma = gamma_conv(act)
        beta = beta_conv(act)
        return gamma, beta

    def forward(self, x, ref):
        normalized_input = self.param_free_norm(x)
        LR, MASK, SF = ref.split([3, 19, 3], dim=1)
        ref_LRM = torch.cat((LR, MASK), dim=1)
        ref_SFM = torch.cat((SF, MASK), dim=1)

        if x.shape[-1] != ref.shape[-1]:
            ref_LRM = nn.functional.interpolate(ref_LRM, x.shape[2:], mode='bicubic', align_corners=False)  # 双三次插值下采样
            ref_SFM = nn.functional.interpolate(ref_SFM, x.shape[2:], mode='bicubic', align_corners=False)  # 双三次插值下采样

        if self.norm_type == 'spade':
            gamma_DI, beta_DI= self.get_gamma_beta(ref_LRM, self.conv1, self.gamma_DI, self.beta_DI)
            gamma_SF, beta_SF = self.get_gamma_beta(ref_SFM, self.conv2, self.gamma_SF, self.beta_SF)

            gamma_JSFT = gamma_SF * (1+gamma_DI) + beta_DI
            beta_JSFT  = beta_SF  * (1+gamma_DI) + beta_DI

            return normalized_input * gamma_JSFT + beta_JSFT

        elif self.norm_type == 'in':
            return normalized_input


class Generator_jsfmgan(nn.Module):
    def __init__(self, input_nc, output_nc, in_size=512, out_size=512, min_feat_size=16, ngf=32, n_blocks=9,
                 parse_ch=19, relu_type='relu',
                 ch_range=[32, 1024], norm_type='spade'):  # 3,3
        super().__init__()

        min_ch, max_ch = ch_range  # 32:1024
        self.min_feat_size = min_feat_size
        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        get_ch = lambda size: ch_clip(1024 * 16 // size)

        self.const_input = nn.Parameter(torch.randn(1, get_ch(min_feat_size), min_feat_size, min_feat_size))

        up_steps = int(np.log2(out_size // min_feat_size))  # 默认up_steps=5
        self.up_steps = up_steps
        ref_ch = parse_ch + 3

        head_ch = get_ch(min_feat_size)  # 默认1024
        head = [
            nn.Conv2d(head_ch, head_ch, kernel_size=(3, 3), padding=1),
            STResBlock(head_ch, head_ch, ref_ch, relu_type, norm_type)
        ]

        body = []

        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            # 小尺寸的body组成
            if i < up_steps - 3:
                body += [
                    nn.Sequential(
                        # nn.Upsample(scale_factor=2),
                        # nn.Conv2d(cin, cout, kernel_size=(3, 3), padding=1),
                        USFMResBlock(cin, rate=2, uptype='pixelshuffle', ref_nc=ref_ch + 3, norm_type=norm_type,
                                     relu_type=relu_type),
                        nn.Conv2d(cout//2, cout, kernel_size=(3, 3), padding=1),
                        STResBlock(cout, cout, ref_ch, relu_type, norm_type)
                    )
                ]
            # 大尺寸的body组成
            else:
                body += [
                    nn.Sequential(
                        USFMResBlock(cin, rate=2, uptype='usfm', ref_nc=ref_ch + 3, norm_type=norm_type,
                                     relu_type=relu_type),
                        nn.Conv2d(cout, cout, kernel_size=(3, 3), padding=1),
                        JSFTResBlock(cout, cout, ref_ch + 3, relu_type, norm_type)
                    )
                ]
            head_ch = head_ch // 2

        self.img_out = nn.Conv2d(ch_clip(head_ch), output_nc, kernel_size=3, padding=1)
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

        self.pyramid_1 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.pyramid_2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.pyramid_3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.pyramid = nn.ModuleList([self.pyramid_1, self.pyramid_2, self.pyramid_3])
        self.temp_feat = None

    def forward_spade(self, net, x, ref):
        for m in net:
            x = self.forward_spade_m(m, x, ref)
        return x

    def forward_spade_m(self, m, x, ref):
        if isinstance(m, STNorm) or isinstance(m, STResBlock) \
                or isinstance(m, JSFT) or isinstance(m, JSFTResBlock) or isinstance(m, AMM) \
                or isinstance(m, USFM) or isinstance(m, USFMResBlock):
            x = m(x, ref)
        else:
            x = m(x)
        return x

    def forward(self, x, ref):
        b, c, h, w = x.shape
        const_input = self.const_input.repeat(b, 1, 1, 1)
        stage_features = []
        # 退化图像拼接人脸解析图
        if ref is not None:
            ref_input = torch.cat((x, ref), dim=1)
        else:
            ref_input = x
        # head
        feat = self.forward_spade(self.head, const_input, ref_input)

        for idx, m in enumerate(self.body):
            # 小尺寸阶段，仅使用退化图像 + 人脸解析图
            if idx <= 1:
                feat = self.forward_spade(m, feat, ref_input)
                # 保存阶段特征
                if idx == 1:
                    self.temp_feat = self.pyramid[idx-1](feat)
                    stage_features.append(self.temp_feat)

            # 大尺寸阶段，仅使用退化图像 + 人脸解析图 + 阶段特征
            else:
                if 1 < idx < 3:
                    if self.temp_feat.shape[-1] != ref_input.shape[-1]:
                        ref_input_temp = nn.functional.interpolate(ref_input, self.temp_feat.shape[2:],
                                                                   mode='bicubic',align_corners=False)
                    else:
                        ref_input_temp = ref_input

                    # 拼接退化图像 + 人脸解析图 + 阶段特征
                    ref_input_temp = torch.cat((ref_input_temp, self.temp_feat), dim=1)
                    feat = self.forward_spade(m, feat, ref_input_temp)

                    # 保存阶段特征
                    self.temp_feat = self.pyramid[idx-1](feat)
                    stage_features.append(self.temp_feat)
                else:
                    if self.temp_feat.shape[-1] != ref_input.shape[-1]:
                        ref_input_temp = nn.functional.interpolate(ref_input, self.temp_feat.shape[2:],
                                                                   mode='bicubic',align_corners=False)
                    else:
                        ref_input_temp = ref_input
                    ref_input_temp = torch.cat((ref_input_temp, self.temp_feat), dim=1)
                    feat = self.forward_spade(m, feat, ref_input_temp)
                    # 保存阶段特征
                    if idx == 3:
                        self.temp_feat = self.pyramid[idx - 1](feat)
                        stage_features.append(self.temp_feat)
        out_img = self.img_out(feat)
        return out_img, stage_features



class Generator_jsfmgan_simple(nn.Module):
    def __init__(self, input_nc, output_nc, in_size=512, out_size=512, min_feat_size=16, ngf=32, n_blocks=9,
                 parse_ch=19, relu_type='relu',
                 ch_range=[32, 512], norm_type='spade'):  # 3,3
        super().__init__()

        min_ch, max_ch = ch_range  # 32:1024
        self.min_feat_size = min_feat_size
        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        get_ch = lambda size: ch_clip(1024 * 16 // size)

        self.const_input = nn.Parameter(torch.randn(1, get_ch(min_feat_size), min_feat_size, min_feat_size))

        up_steps = int(np.log2(out_size // min_feat_size))  # 默认up_steps=5
        self.up_steps = up_steps
        ref_ch = parse_ch + 3

        head_ch = get_ch(min_feat_size)  # 默认1024
        head = [
            nn.Conv2d(head_ch, head_ch, kernel_size=(3, 3), padding=1),
            STResBlock(head_ch, head_ch, ref_ch, relu_type, norm_type)
        ]

        body = []

        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            # 小尺寸的body组成
            if i < up_steps - 3:
                body += [
                    nn.Sequential(
                        # nn.Upsample(scale_factor=2),
                        # nn.Conv2d(cin, cout, kernel_size=(3, 3), padding=1),
                        USFMResBlock(cin, rate=2, uptype='pixelshuffle', ref_nc=ref_ch + 3, norm_type=norm_type,
                                     relu_type=relu_type),
                        nn.Conv2d(cout//2, cout, kernel_size=(3, 3), padding=1),
                        STResBlock(cout, cout, ref_ch, relu_type, norm_type)
                    )
                ]
            # 大尺寸的body组成
            else:
                body += [
                    nn.Sequential(
                        USFMResBlock(cin, rate=2, uptype='usfm', ref_nc=ref_ch + 3, norm_type=norm_type,
                                     relu_type=relu_type),
                        nn.Conv2d(cout, cout, kernel_size=(3, 3), padding=1),
                        JSFTResBlock(cout, cout, ref_ch + 3, relu_type, norm_type)
                    )
                ]
            head_ch = head_ch // 2

        self.img_out = nn.Conv2d(ch_clip(head_ch), output_nc, kernel_size=3, padding=1)
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

        self.pyramid_1 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.pyramid_2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.pyramid_3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.pyramid = nn.ModuleList([self.pyramid_1, self.pyramid_2, self.pyramid_3])
        self.temp_feat = None

    def forward_spade(self, net, x, ref):
        for m in net:
            x = self.forward_spade_m(m, x, ref)
        return x

    def forward_spade_m(self, m, x, ref):
        if isinstance(m, STNorm) or isinstance(m, STResBlock) \
                or isinstance(m, JSFT) or isinstance(m, JSFTResBlock) or isinstance(m, AMM) \
                or isinstance(m, USFM) or isinstance(m, USFMResBlock):
            x = m(x, ref)
        else:
            x = m(x)
        return x

    def forward(self, x, ref):
        b, c, h, w = x.shape
        const_input = self.const_input.repeat(b, 1, 1, 1)
        stage_features = []
        # 退化图像拼接人脸解析图
        if ref is not None:
            ref_input = torch.cat((x, ref), dim=1)
        else:
            ref_input = x
        # head
        feat = self.forward_spade(self.head, const_input, ref_input)

        for idx, m in enumerate(self.body):
            # 小尺寸阶段，仅使用退化图像 + 人脸解析图
            if idx <= 1:
                feat = self.forward_spade(m, feat, ref_input)
                # 保存阶段特征
                if idx == 1:
                    self.temp_feat = self.pyramid[idx-1](feat)
                    stage_features.append(self.temp_feat)

            # 大尺寸阶段，仅使用退化图像 + 人脸解析图 + 阶段特征
            else:
                if 1 < idx < 3:
                    if self.temp_feat.shape[-1] != ref_input.shape[-1]:
                        ref_input_temp = nn.functional.interpolate(ref_input, self.temp_feat.shape[2:],
                                                                   mode='bicubic',align_corners=False)
                    else:
                        ref_input_temp = ref_input

                    # 拼接退化图像 + 人脸解析图 + 阶段特征
                    ref_input_temp = torch.cat((ref_input_temp, self.temp_feat), dim=1)
                    feat = self.forward_spade(m, feat, ref_input_temp)

                    # 保存阶段特征
                    self.temp_feat = self.pyramid[idx-1](feat)
                    stage_features.append(self.temp_feat)
                else:
                    if self.temp_feat.shape[-1] != ref_input.shape[-1]:
                        ref_input_temp = nn.functional.interpolate(ref_input, self.temp_feat.shape[2:],
                                                                   mode='bicubic',align_corners=False)
                    else:
                        ref_input_temp = ref_input
                    ref_input_temp = torch.cat((ref_input_temp, self.temp_feat), dim=1)
                    feat = self.forward_spade(m, feat, ref_input_temp)
                    # 保存阶段特征
                    if idx == 3:
                        self.temp_feat = self.pyramid[idx - 1](feat)
                        stage_features.append(self.temp_feat)
        out_img = self.img_out(feat)
        return out_img, stage_features