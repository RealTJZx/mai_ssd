# Readme
"""
Author: xYH
Create_time : 2024-02-21-12:35
    Desc:
    TIP:
        1.
        2.
"""
import torch
from torch import nn



def conv2d_bn_active(x, in_channels, out_channels, kernel_size_1, kernel_size_2, padding=None, stride=1, dilation_rate=1, relu=True):
    x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size_1, kernel_size_2), padding=(kernel_size_1//2 if not padding else padding, kernel_size_2//2 if not padding else padding), stride=(stride, stride), dilation=dilation_rate, bias=False)(x)
    x = nn.BatchNorm2d(out_channels)(x)
    if relu:
        x = nn.ReLU()(x)
    return x

def BasicRFB(x, in_channels, out_channels, stride=1, map_reduce=8):
    mid_channels = in_channels // map_reduce

    branch_0 = conv2d_bn_active(x, in_channels, mid_channels*2, kernel_size_1=1, kernel_size_2=1, stride=stride)
    branch_0 = conv2d_bn_active(branch_0, mid_channels*2, mid_channels*2, kernel_size_1=3, kernel_size_2=3, relu=False)

    branch_1 = conv2d_bn_active(x, in_channels, mid_channels, kernel_size_1=1, kernel_size_2=1)
    branch_1 = conv2d_bn_active(branch_1, mid_channels, mid_channels*2, kernel_size_1=3, kernel_size_2=3, stride=stride)
    branch_1 = conv2d_bn_active(branch_1, mid_channels*2, mid_channels*2, kernel_size_1=3, kernel_size_2=3, padding=3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn_active(x, in_channels, mid_channels, kernel_size_1=1, kernel_size_2=1)
    branch_2 = conv2d_bn_active(branch_2, mid_channels, (mid_channels//2)*3, kernel_size_1=3, kernel_size_2=3)
    branch_2 = conv2d_bn_active(branch_2, (mid_channels//2)*3, mid_channels*2, kernel_size_1=3, kernel_size_2=3, stride=stride)
    branch_2 = conv2d_bn_active(branch_2, mid_channels*2, mid_channels*2, kernel_size_1=3, kernel_size_2=3, padding=5, dilation_rate=5, relu=False)

    branch_3 = conv2d_bn_active(x, in_channels, mid_channels, kernel_size_1=1, kernel_size_2=1)
    branch_3 = conv2d_bn_active(branch_3, mid_channels, (mid_channels//2)*3, kernel_size_1=1, kernel_size_2=7)
    branch_3 = conv2d_bn_active(branch_3, (mid_channels//2)*3, mid_channels*2, kernel_size_1=7, kernel_size_2=1, stride=stride)
    branch_3 = conv2d_bn_active(branch_3, mid_channels*2, mid_channels*2, kernel_size_1=3, kernel_size_2=3, padding=7, dilation_rate=7, relu=False)

    # zx
    # 小分支的最后输出没有使用激活；
    # 小分支在 concat 后调整通道时也没有激活
    # 小分支与直连相加时，才有的激活
    out = torch.concat([branch_0, branch_1, branch_2, branch_3], dim=1)
    out = conv2d_bn_active(out, in_channels, out_channels, kernel_size_1=1, kernel_size_2=1, relu=False)

    short = conv2d_bn_active(x, in_channels, out_channels, kernel_size_1=1, kernel_size_2=1, stride=stride, relu=False)
    out = out + short
    out = nn.ReLU()(out)
    return out



if __name__ == "__main__":
    test_data = torch.randn((8, 128, 300, 300))
    test_out = BasicRFB(test_data, 128, 256)
    print(test_out.shape)
