# Readme
"""
Author: xYH
Create_time : 2024-03-10-19:14
    Desc:
        常规的注意力机制
    TIP:
        1.
        2.
"""
import torch
from torch import nn
import math





# SENet
class SENet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel//ratio,bias=False),
            nn.BatchNorm1d(channel//ratio),         # 后加
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            nn.BatchNorm1d(channel),                # 后加
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, h, w = x.size()
        # print("inputs_shape:", x.shape)
        avg = self.avg_pool(x).view([b, c])
        # print("avg_shape:", avg.shape)
        fc = self.fc(avg).view([b, c, 1, 1])
        # print("断点_FC后的值：", fc)
        # print("fc_shape:", fc.shape)
        return x * fc

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ratio, False),
            # nn.BatchNorm1d(channel//ratio),     # 后加
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            # nn.BatchNorm1d(channel // ratio),   # 后加
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        # 共享全连接层的处理
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class SpacialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpacialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        # print("max_pool_out_shape:", np.shape(max_pool_out))
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        # print("avg_pool_out_shape:", np.shape(avg_pool_out))
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        outputs = self.conv(pool_out)
        outputs = self.sigmoid(outputs)
        return outputs * x

class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spacial_attention = SpacialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x


# ECANet
class ECANet_Block(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECANet_Block, self).__init__()
        # 根据输入通道，自适应调整卷积核大小
        kernel_size = int(abs((math.log(channel, 2)+ b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        padding = kernel_size // 2
        self.conv_1d = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x)
        # print("变形前的平均池化后特征层形状：", avg.size())
        avg = avg.view([b, 1, c])
        out = self.conv_1d(avg)
        # print("变形前的卷积后特征层形状：", out.size())
        out = self.sigmoid(out).view([b, c, 1, 1])
        # print("该注意力机制的输出为：", out)
        return out * x