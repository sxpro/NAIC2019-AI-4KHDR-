## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
def default_conv(in_channels, out_channels, kernel_size, bias=True,groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), groups=groups, bias=bias)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        self.res_scale1 = nn.Parameter(torch.ones(1))
        self.res_scale2 = nn.Parameter(torch.ones(1))
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = self.res_scale1 * res + self.res_scale2 * x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, bn=False):
        super(ResidualGroup, self).__init__()
        self.res_scale1 = nn.Parameter(torch.ones(1))
        self.res_scale2 = nn.Parameter(torch.ones(1))
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.res_scale1 * res + self.res_scale2 * x
        return res

class ColorNet(nn.Module):
    def __init__(self, conv=default_conv):
        super(ColorNet, self).__init__()

        n_resgroups = 2
        n_resblocks = 3
        n_feats = 16
        kernel_size = 3
        reduction = 8
        res_scale = 0.2

        act = nn.LeakyReLU(0.1, True)

        modules_head = [conv(3, n_feats, kernel_size=3)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks, bn=False) \
            for _ in range(n_resgroups)]

        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            act,
            conv(n_feats, 3, kernel_size)]

        # self.add_mean = MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
