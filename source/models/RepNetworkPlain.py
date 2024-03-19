import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import imp
import torch.nn as nn
import torch.nn.functional as F
from source.models.convnet_utils import conv_bn, conv_bn_relu

from source.models.basic import ConvBN
from source.models.blocks import DBB, OREPA_1x1, OREPA, OREPA_LargeConvBase, OREPA_LargeConv
from source.models.blocks_repvgg import RepVGGBlock, RepVGGBlock_OREPA

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act  = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
            
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y
               
class RepNetwork_V001(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(RepNetwork_V001, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        self.scale_in_network = 3

        self.pixelUnShuffle = nn.PixelUnshuffle(self.scale_in_network)
        
        backbone = []
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None)
        self.head = conv_bn(in_channels=self.colors * (self.scale_in_network * self.scale_in_network), out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)
        
        for i in range(self.module_nums):
            backbone += [conv_bn(in_channels=self.channel_nums, out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(conv_bn(in_channels=self.channel_nums, out_channels=self.colors*self.scale*self.scale*(self.scale_in_network*self.scale_in_network), kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA))
                                        #torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        #conv_bn(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale*self.scale_in_network)
        #self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y + y0)
        #y = torch.clip(y, 0, 1.)
        y = self.upsampler(y) 
        return y
    
    def fuse_model(self):
        pass
    
class RepNetwork_V002(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(RepNetwork_V002, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        self.scale_in_network = 3

        self.pixelUnShuffle = nn.PixelUnshuffle(self.scale_in_network)
        
        backbone = []
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None)
        self.head = conv_bn(in_channels=self.colors * (self.scale_in_network * self.scale_in_network), out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)
        
        for i in range(self.module_nums):
            backbone += [conv_bn(in_channels=self.channel_nums, out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(conv_bn(in_channels=self.channel_nums, out_channels=self.colors*self.scale*self.scale*(self.scale_in_network*self.scale_in_network), kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type='linear', assign_type=OREPA))
                                        #torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        #conv_bn(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale*self.scale_in_network)
        #self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0)
        
        #y = torch.cat([y, x], dim=1)
        y = self.transition(y + y0)
        #y = torch.clip(y, 0, 1.)
        y = self.upsampler(y)
        return y
    
    def fuse_model(self):
        pass
    
class RepNetwork_V003(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(RepNetwork_V003, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.scale_in_network = 3

        backbone = []
        self.pixelUnShuffle = nn.PixelUnshuffle(self.scale_in_network)
        
        self.head = conv_bn(in_channels=self.colors * (self.scale_in_network * self.scale_in_network), out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)
        
        for i in range(self.module_nums):
            backbone += [conv_bn(in_channels=self.channel_nums, out_channels=self.channel_nums, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type=act_type, assign_type=OREPA)]
        
        self.backbone = nn.Sequential(*backbone)
    
        self.transition = nn.Sequential(conv_bn(in_channels=self.channel_nums, out_channels=self.colors*self.scale*self.scale*(self.scale_in_network*self.scale_in_network), kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type='linear', assign_type=OREPA))
        self.upsampler = nn.PixelShuffle(self.scale*self.scale_in_network)
    
    def forward(self, x):
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0)
        
        y = self.transition(y + y0)
        #y = torch.clip(y, 0, 1.)
        y = self.upsampler(y)
        return y
    
    def fuse_model(self):
        pass
        
class PlainRepConv_BlockV2_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainRepConv_BlockV2_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
        y0 = self.head(x)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y + y0)
        #y = torch.clip(y,0,255.)
        y = self.upsampler(y) 
        return y
    
if __name__ == "__main__":
    x = torch.rand(1,3,128,128).cuda()
    model = PlainRepConv(module_nums=6, channel_nums=64, act_type='prelu', scale=3, colors=3).cuda().eval()
    y0 = model(x)

    model.fuse_model()
    y1 = model(x)

    print(model)
    print(y0-y1)
    print('->Matching Error: {}'.format(np.mean((y0.detach().cpu().numpy() - y1.detach().cpu().numpy()) ** 2)))    # Will be around 1e-10