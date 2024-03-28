# Readme
"""
Author: xYH
Create_time : 2024-03-10-21:09
    Desc:
    TIP:
        1.
        2.
"""
import torch
from torch import nn

from nets.zx.parts.x_base import BaseConv, DWConv, CSPLayer, Transition, SppCsp
from nets.zx.parts.fpn_for_cspz import FPN
from nets.zx.parts.attention_module import AST





class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, depthwise=False, act="silu", use_fpn=False, use_attention=False):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv
        self.use_attention = use_attention
        self.use_fpn = use_fpn

        base_channels   = int(wid_mul * 64)  # 32
        self.base_channels = base_channels
        base_depth      = max(round(dep_mul * 3), 1)  # 1

        # 代表每个大结构块输出的特征层的通道数；倒数 6 个为进入检测头的特征层的通道数
        # [32, 64, 128, 256, 512, 256, 256, 256]
        self.feature_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8,
                                 base_channels * 16, base_channels * 8, base_channels * 8, base_channels * 8]

        # 初步特征提取结构块
        # 480,480,3 ==> 480,480,16 -> 240,240,32
        pre_base_channels = base_channels // 2
        self.stem = nn.Sequential(
            # zx - 480, 480, 16
            Conv(3, pre_base_channels, 3, 1, act=act),
            # zx - 240, 240, 32
            Conv(pre_base_channels, base_channels, 3, 2, act=act),
        )

        # 特征提取结构块，初次使用到 CSPLayer 进行特征提取
        # 240,240,32 ==> 120,120,64 -> 120,120,64 -> 120,120,128
        self.dark2 = nn.Sequential(
            # zx - 120, 120, 64
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, act=act),
            # zx - 120, 120, 128
            CSPLayer(base_channels * 2, base_channels * 4, n=base_depth * 3, act=act),
        )

        # 进入主要特征提取部分，从这部分开始输出有效特征层
        # 120,120,128 ==> 60,60,128 -> 60,60,256
        self.dark3 = nn.Sequential(
            # zx - 60, 60, 128
            Transition(base_channels * 4),
            BaseConv(base_channels*4, base_channels*4, 1, 1) if self.use_attention else nn.Identity(),
            AST(base_channels * 4, kernel_size=9, testing_flag=False) if self.use_attention else nn.Identity(),
            # zx - 60, 60, 256
            CSPLayer(base_channels * 4, base_channels * 8, n=base_depth * 3, act=act),
        )

        #  60,60,256 ==> 30,30,256 -> 30,30,512
        self.dark4 = nn.Sequential(
            # zx - 30, 30, 256
            Transition(base_channels * 8),
            BaseConv(base_channels * 8, base_channels * 8, 1, 1) if self.use_attention else nn.Identity(),
            AST(base_channels * 8, kernel_size=7, testing_flag=False) if self.use_attention else nn.Identity(),
            # zx - 30, 30, 512
            CSPLayer(base_channels * 8, base_channels * 16, n=base_depth * 3, act=act),
        )

        # 使用到了 SPP
        # 30,30,512 ==> 15,15,512 -> 15,15,512 -> 15,15,512
        self.dark5 = nn.Sequential(
            # zx - 15, 15, 512
            Transition(base_channels * 16),
            BaseConv(base_channels * 16, base_channels * 16, 1, 1) if self.use_attention else nn.Identity(),
            AST(base_channels * 16, kernel_size=5, testing_flag=False) if self.use_attention else nn.Identity(),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, act=act),
            SppCsp(in_channels=base_channels * 16, out_channels=base_channels * 16),
        )

        # 输出的有效特征层需要通道减半；
        self.conv11_for_original_feat3 = BaseConv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv11_for_original_feat4 = BaseConv(base_channels * 16, base_channels * 8, 1, 1)

        if self.use_fpn:
            # print("（定义）使用了FPN！")
            if self.use_attention:
                self.fpn = FPN(base_channels=base_channels, base_depth=base_depth, use_att=True)
                self.conv11_for_att_out5 = BaseConv(base_channels * 16, base_channels * 16, 1, 1)
                self.att_for_out5 = AST(base_channels * 16, testing_flag=False)
            else:
                self.fpn = FPN(base_channels=base_channels, base_depth=base_depth)
            self.conv11_bt_feat5_out5 = BaseConv(base_channels * 16, base_channels * 8, 1, 1)
            self.feat5_conv_down = Transition(base_channels * 8, max_pool_ceil=True)
            self.conv11_for_out5 = BaseConv(base_channels * 16, 128, 1, 1)
        else:
            # 8,8,256 ==> 8,8,128
            self.conv11_for_out5 = BaseConv(base_channels * 8, 128, 1, 1)

        # zx - 后三层输出的处理
        # 15,15,512 ==> 15,15,128 -> 8,8,256
        self.conv11_for_out4 = BaseConv(base_channels * 16, 128, 1, 1)
        self.conv33_for_out4 = BaseConv(128, base_channels * 8, 3, 2)

        # 8,8,256 ==> 6,6,256
        self.conv33_for_out5 = BaseConv(128, base_channels * 8, 3, 1, padding=0)

        # 6,6,256 ==> 6,6,128 -> 4,4,256
        self.conv11_for_out6 = BaseConv(base_channels * 8, 128, 1, 1)
        self.conv33_for_out6 = BaseConv(128, base_channels * 8, 3, 1, padding=0)

    def forward(self, x):
        # 480,480,3 ==> 240,240,32
        x = self.stem(x)
        # 240,240,32 ==> 120,120,128
        x = self.dark2(x)

        # 120,120,128 ==> 60,60,256
        feat3_ = self.dark3(x)
        # 60,60,256 ==> 60,60,128
        feat3_out_1 = self.conv11_for_original_feat3(feat3_)
        # 60,60,256 ==> 30,30,512
        feat4_ = self.dark4(feat3_)
        # 30,30,512 ==> 30,30,256
        feat4_out_2 = self.conv11_for_original_feat4(feat4_)
        # 30,30,512 ==> 15,15,512
        feat5_out_3 = self.dark5(feat4_)

        if self.use_fpn:
            # feat3_out_1: 60,60,128
            # feat4_out_2: 30,30,256
            # feat5_out_3: 15,15,512
            feat3_out_1, feat4_out_2, feat5_out_3 = self.fpn(feat3_out_1, feat4_out_2, feat5_out_3)

            # 15,15,512 ==> 15,15,128 -> 8,8,256
            out_4 = self.conv11_for_out4(feat5_out_3)
            out_4 = self.conv33_for_out4(out_4)

            # 15,15,512 ==> 15,15,256 -> 8,8,256 -> 8,8,512
            out_5_bf = self.conv11_bt_feat5_out5(feat5_out_3)
            out_5_bf = self.feat5_conv_down(out_5_bf)
            out_5_bf = torch.cat([out_5_bf, out_4], dim=1)

            if self.use_attention:
                out_5_bf = self.conv11_for_att_out5(out_5_bf)
                out_5_bf = self.att_for_out5(out_5_bf)

            out_5 = self.conv11_for_out5(out_5_bf)
        else:
            out_4 = self.conv11_for_out4(feat5_out_3)
            out_4 = self.conv33_for_out4(out_4)
            out_5 = self.conv11_for_out5(out_4)

        out_5 = self.conv33_for_out5(out_5)
        out_6 = self.conv11_for_out6(out_5)
        out_6 = self.conv33_for_out6(out_6)


        return feat3_out_1, feat4_out_2, feat5_out_3, out_4, out_5, out_6





if __name__ == "__main__":
    net_csp = CSPDarknet(0.33, 0.5)
    net_csp_fpn = CSPDarknet(0.33, 0.5, use_fpn=True)
    net_csp_fpn_at = CSPDarknet(0.33, 0.5, use_fpn=True, use_attention=True)

    testDict = {
        net_csp: "正常的 cspdarknet ",
        net_csp_fpn: "使用 fpn ",
        net_csp_fpn_at: "使用 fpn + attention ",
    }

    inData = torch.randn((4, 3, 480, 480))
    for testingNet in [net_csp, net_csp_fpn, net_csp_fpn_at]:
        feat3_out_1, feat4_out_2, feat5_out_3, out_4, out_5, out_6 = testingNet(inData)
        print(f"{testDict[testingNet]} 形状是：")
        print(feat3_out_1.shape)
        print(feat4_out_2.shape)
        print(feat5_out_3.shape)
        print(out_4.shape)
        print(out_5.shape)
        print(out_6.shape)
