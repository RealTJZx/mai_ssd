# Readme
"""
Author: xYH
Create_time : 2024-03-10-20:11
    Desc:
    TIP:
        1.
        2.
"""
import torch
from torch import nn

from nets.zx.parts.x_base import BaseConv, CSPLayer, Transition
from nets.zx.parts.attention_module import AST





class FPN(nn.Module):
    def __init__(self, base_channels, base_depth, use_att=False):
        super(FPN, self).__init__()

        # 32: 15, 15, 512 -> 15, 15, 256
        self.feat5_1X1_conv = BaseConv(base_channels * 16, base_channels * 8, 1, 1)

        # 32: 15, 15, 256 -> 30, 30, 256
        self.upsample_1 = nn.Upsample(scale_factor=2,mode="nearest")
        # feat5 上采样后连接后
        # 32: 30, 30, 512 -> 30, 30, 256
        self.feat4_C = CSPLayer(base_channels * 16, base_channels * 8, base_depth, shortcut=False)
        # 32: 30, 30, 256 -> 30, 30, 128
        self.feat4_C_1x1 = BaseConv(base_channels * 8, base_channels * 4, 1, 1)

        # 32: 30, 30, 128 -> 60, 60, 128
        self.upsample_2 = nn.Upsample(scale_factor=2,mode="nearest")
        # feat4 上采样后连接后
        # 确定输出 feat3
        # 32: 60, 60, 256 -> 60, 60, 128
        # base_channels * 4 = 128
        self.feat3_C_final = CSPLayer(base_channels * 8, base_channels * 4, base_depth * 3, shortcut=False)

        # 32: 60, 60, 256 -> 30, 30, 256
        self.feat3_conv_down = Transition(base_channels * 4)
        # 确定输出 feat4
        # 32: 30, 30, 512 -> 30, 30, 256
        self.feat4_C_final = CSPLayer(base_channels * 8, base_channels * 8, base_depth * 3, shortcut=False)

        # 32: 30, 30, 256 -> 15, 15, 256
        self.feat4_conv_down = Transition(base_channels * 8)
        # 32: 15, 15, 512 -> 15, 15, 512
        self.feat5_C_final = CSPLayer(base_channels * 16, base_channels * 16, base_depth * 3, shortcut=False)

        self.use_att = use_att
        if self.use_att:
            # print("（定义）FPN 使用了注意力！")
            self.up_conv11_for_feat4 = BaseConv(base_channels*16, base_channels*16, 1, 1)
            self.up_conv11_for_feat3 = BaseConv(base_channels*8, base_channels*8, 1, 1)
            self.dn_conv11_for_feat4 = BaseConv(base_channels*8, base_channels*8, 1, 1)
            self.dn_conv11_for_feat5 = BaseConv(base_channels*16, base_channels*16, 1, 1)

            self.up_att_for_feat4 = AST(base_channels * 16, kernel_size=5)
            self.up_att_for_feat3 = AST(base_channels * 8, kernel_size=7)
            self.dn_att_for_feat4 = AST(base_channels * 8, kernel_size=7)
            self.dn_att_for_feat5 = AST(base_channels * 16, kernel_size=5)
    def forward(self, feat3_in, feat4_in, feat5_in):
        # # 15, 15, 512   ->  15, 15, 256
        feat5_bf_up = self.feat5_1X1_conv(feat5_in)

        # 15, 15, 256   ->  30, 30, 256
        feat5_up = self.upsample_1(feat5_bf_up)
        # 30, 30, 256   ->   30, 30, 512
        feat4_bf_up = torch.cat([feat5_up, feat4_in], dim=1)
        if self.use_att:
            feat4_bf_up = self.up_conv11_for_feat4(feat4_bf_up)
            feat4_bf_up = self.up_att_for_feat4(feat4_bf_up)
        # 30, 30, 512   ->   30, 30, 256
        feat4_bf_up = self.feat4_C(feat4_bf_up)
        # 30, 30, 256   ->   30, 30, 128
        feat4_bf_up = self.feat4_C_1x1(feat4_bf_up)

        # 30, 30, 128   ->   60, 60, 128
        feat4_up = self.upsample_2(feat4_bf_up)
        # 60, 60, 128   ->   60, 60, 256
        feat3_bf_dn = torch.cat([feat4_up, feat3_in], dim=1)
        if self.use_att:
            feat3_bf_dn = self.up_conv11_for_feat3(feat3_bf_dn)
            feat3_bf_dn = self.up_att_for_feat3(feat3_bf_dn)
        # 60, 60, 256   ->   60, 60, 128
        feat3_bf_dn = self.feat3_C_final(feat3_bf_dn)
        # 60, 60, 128   ->   60, 60, 128
        feat3_out_1 = feat3_bf_dn
        feat3_dn = feat3_bf_dn

        # 60, 60, 128   ->   30, 30, 128
        feat3_dn = self.feat3_conv_down(feat3_dn)
        # 30, 30, 128   ->   30, 30, 256
        feat4_in_dn = torch.cat([feat3_dn, feat4_bf_up], dim=1)
        if self.use_att:
            feat4_in_dn = self.dn_conv11_for_feat4(feat4_in_dn)
            feat4_in_dn = self.dn_att_for_feat4(feat4_in_dn)
        # 30, 30, 256   ->   30, 30, 256
        feat4_dn = self.feat4_C_final(feat4_in_dn)
        feat4_out_2 = feat4_dn

        # 30, 30, 256   ->   15, 15, 256
        feat4_dn = self.feat4_conv_down(feat4_dn)
        # 15, 15, 256   ->   15, 15, 512
        feat5_out_3 = torch.cat([feat4_dn, feat5_bf_up], dim=1)
        if self.use_att:
            # print("（传播）FPN 使用了注意力！")
            feat5_out_3 = self.dn_conv11_for_feat5(feat5_out_3)
            feat5_out_3 = self.dn_att_for_feat5(feat5_out_3)
        feat5_out_3 = self.feat5_C_final(feat5_out_3)

        return feat3_out_1, feat4_out_2, feat5_out_3





if __name__ == "__main__":
    feat3 = torch.randn((4, 128, 60, 60))
    feat4 = torch.randn((4, 256, 30, 30))
    feat5 = torch.randn((4, 512, 15, 15))

    fpnnet = FPN(32, 1, use_att=True)
    out3, out4, out5 = fpnnet(feat3, feat4, feat5)

    print("输出形状对应为：")
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)