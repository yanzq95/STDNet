import torch
from torch import nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
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
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def compute_interframe_diff(deps):

    n, t, c, h, w = deps.size()  ## same resolution as final output

    deps_1 = deps[:, :t - 1, :, :, :]
    deps_2 = deps[:, 1:, :, :, :]

    deps_3 = deps[:, :t - 2, :, :, :]
    deps_4 = deps[:, 2:, :, :, :]

    deps_1 = deps_1.reshape(-1, c, h, w)
    deps_2 = deps_2.reshape(-1, c, h, w)
    diff_maps = torch.abs(deps_1 - deps_2)

    diff_maps = diff_maps.view(n, t - 1, c, h, w)

    deps_3 = deps_3.reshape(-1, c, h, w)
    deps_4 = deps_4.reshape(-1, c, h, w)
    diff_maps_cross = torch.abs(deps_4 - deps_3)

    diff_maps_cross = diff_maps_cross.view(n, t - 2, c, h, w)

    return diff_maps, diff_maps_cross