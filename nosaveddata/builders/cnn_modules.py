import torch
from torch.nn import functional as F
from torch import nn



# Rethinking Atrous Convolution for Semantic Image Segmentation
class ASPP(nn.Module):
    def __init__(self, channels, atrous_rates=(2,4), ks=3):
        super(ASPP, self).__init__()

        self.convs = nn.ModuleList()
        for rate in atrous_rates:
          self.convs.append(nn.Sequential(nn.Conv2d(channels, channels, ks, dilation=rate, padding=(ks//2+rate-1)),
                                          nn.BatchNorm2d(channels)))

        self.conv_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, channels, 1),
                                       nn.BatchNorm2d(channels))

        expand_ratio = len(atrous_rates) + 1
        self.out_conv = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(channels*expand_ratio, channels, 1),
                                      nn.BatchNorm2d(channels),
                                      nn.ReLU(True))

    def forward(self,x):

        xs = []

        for conv in self.convs:
          _x = conv(x)
          xs.append(_x)
        
        _x = self.conv_pool(x)
        upsampled = F.interpolate(_x, size=x.shape[-2:], mode='bilinear', align_corners=False)
        xs.append(upsampled)

        x = torch.cat(xs,-3)
        
        x = self.out_conv(x)
        
        return x