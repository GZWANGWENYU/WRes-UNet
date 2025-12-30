""" Parts of the U-Net model """
# 小波深度可分离卷积Unet

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from WTConv import WTConv2d

class SmartShortcut(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            # 可选添加 动态通道加权
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 16, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch // 16, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.wtconv1 = nn.Sequential(
            WTConv2d(in_channels, in_channels, kernel_size=3, bias=False,wt_levels=5),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.wtconv2 = nn.Sequential(
            WTConv2d(mid_channels, out_channels, kernel_size=3, bias=False, wt_levels=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bat = nn.BatchNorm2d(out_channels)
        self.shortcut = SmartShortcut(in_channels,out_channels,1)



        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        #print("residual:", residual.size())
        x1 = self.wtconv1(x)
        x1 = self.wtconv2(x1)
        #print("x1:",x1.size())
        #return self.relu(self.bat(residual + x1))
        #print(x1)
        #print(residual)
        #print(residual+x1)
        return self.relu(residual + x1)
        #return x1

class DoubleConv1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WRes_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WRes_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv1(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def save_model_info(model, input_size=(1, 1, 512, 512), txt_file="model_info.txt"):
    dummy_input = torch.randn(input_size)
    flops, params = profile(model, inputs=(dummy_input,))
    with open(txt_file, 'w') as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"FLOPs: {flops}\n")

if __name__ =="__main__":

    net=WRes_UNet(1,1)
    save_model_info(net)

    #print(net)
    input = torch.randn(1, 1, 256, 256)  #B C H W
    output = net(input)
    print(output.size())