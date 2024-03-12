import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from source.models.RepConv_block import RepBlock, RepBlockV2, RepBlockV3
except ModuleNotFoundError:
    from RepConv_block import RepBlock, RepBlockV2

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
    

class DownRepConv_Block(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(DownRepConv_Block, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.pixelUnShuffle = nn.PixelUnshuffle(3)
        self.head = RepBlockV2(inp_planes=self.colors*9, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [RepBlockV2(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        RepBlockV2(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale*9, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale*3)
    
    def forward(self, x):
        
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y
    
    def fuse_model(self):
        ## reparam as plainsrcd cd
        for idx, blk in enumerate(self.backbone):
            if type(blk) == RepBlockV2:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone
                self.backbone[idx] = conv.to(RK.device)
        #for idx, blk in enumerate(self.head):
        if type(self.head) == RepBlockV2:
            RK, RB  = self.head.repblock_convert()
            conv = Conv3X3(self.head.inp_planes, self.head.out_planes, act_type=self.head.act_type)
            ## update weights & bias for conv3x3
            conv.conv3x3.weight.data = RK
            conv.conv3x3.bias.data   = RB
            ## update weights & bias for activation
            if self.head.act_type == 'prelu':
                conv.act.weight = self.head.act.weight
            ## update block for backbone
            self.head = conv.to(RK.device)
        for idx, blk in enumerate(self.transition):
            if type(blk) == RepBlockV2:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone
                self.transition[idx] = conv.to(RK.device)


class DownRepConv_Block_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(DownRepConv_Block_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.pixelUnShuffle = nn.PixelUnshuffle(3)

        self.head = Conv3X3(inp_planes=self.colors*9, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale*9, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale*3)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
        y0 = self.pixelUnShuffle(x)

        y0 = self.head(y0)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y