# Readme
"""
Author: xYH
Create_time : 2024-03-11-2:30
    Desc:
    TIP:
        1.
        2.
"""
import torch
from torch import nn



inData = torch.randn((4, 3, 15, 15))
testnet = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False)
outData = testnet(inData)
print(outData.shape)

inData = torch.randn((4, 3, 8, 8))
testnet = nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1, bias=False)
outData = testnet(inData)
print(outData.shape)

inData = torch.randn((4, 3, 6, 6))
testnet = nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1, bias=False)
outData = testnet(inData)
print(outData.shape)