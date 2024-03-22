import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import imp
import torch.nn as nn
import torch.nn.functional as F

try:
    from source.models.RepConv_block import RepBlock, RepBlockV2, RepBlockV3, RepBlockV4, RepBlockV5, RepBlockV6
    from source.models.convnet_utils import conv_bn, conv_bn_relu

    from source.models.basic import ConvBN
    from source.models.blocks import DBB, OREPA_1x1, OREPA, OREPA_LargeConvBase, OREPA_LargeConv
    from source.models.blocks_repvgg import RepVGGBlock, RepVGGBlock_OREPA

except ModuleNotFoundError:
    from RepConv_block import RepBlock, RepBlockV2, RepBlockV3, RepBlockV4, RepBlockV5, RepBlockV6

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
        elif self.act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
class Conv3X3DWS(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3DWS, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.depthwise = torch.nn.Conv2d(self.inp_planes, self.inp_planes, kernel_size=3, padding=1, groups=self.inp_planes)
        self.pointwise = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
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
        elif self.act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(y)
        if self.act_type != 'linear':
            y = self.act(y)
        return y
    

class RepNetwork_V011_BestStruct_teacher(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors, block_type=RepBlockV2):
        super(RepNetwork_V011_BestStruct_teacher, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        
        self.repBlk = block_type

        backbone = []
        backbone2 = []
        self.pixelUnShuffle = nn.PixelUnshuffle(3)
        self.head = self.repBlk(inp_planes=self.colors*9, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [self.repBlk(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type)]
        
        self.backbone = nn.Sequential(*backbone)

        self.transition = nn.Sequential(self.repBlk(inp_planes=self.channel_nums, out_planes=self.colors*9, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(3)
        
        self.head2 = self.repBlk(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)

        for i in range(5):
            backbone2 += [self.repBlk(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone2 = nn.Sequential(*backbone2)

        self.input_conv2 = self.repBlk(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear')

        self.upsampler1 = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y)
        y = self.upsampler(y) 
        y = self.head2(y+x)
        y = self.backbone2(y)
        y = self.input_conv2(y)
        y = self.upsampler1(y) 
        return y
    
    def fuse_model(self):
        ## reparam as plainsrcd cd
        for idx, blk in enumerate(self.backbone):
            if type(blk) == self.repBlk:
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
        if type(self.head) == self.repBlk:
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


        for idx, blk in enumerate(self.backbone2):
            if type(blk) == self.repBlk:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone2
                self.backbone2[idx] = conv.to(RK.device)
        #for idx, blk in enumerate(self.head):
        if type(self.head2) == self.repBlk:
            RK, RB  = self.head2.repblock_convert()
            conv = Conv3X3(self.head2.inp_planes, self.head2.out_planes, act_type=self.head2.act_type)
            ## update weights & bias for conv3x3
            conv.conv3x3.weight.data = RK
            conv.conv3x3.bias.data   = RB
            ## update weights & bias for activation
            if self.head2.act_type == 'prelu':
                conv.act.weight = self.head2.act.weight
            ## update block for backbone
            self.head2 = conv.to(RK.device)


        for idx, blk in enumerate(self.transition):
            if type(blk) == self.repBlk:
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
        if type(self.input_conv2) == self.repBlk:
            RK, RB  = self.input_conv2.repblock_convert()
            conv = Conv3X3(self.input_conv2.inp_planes, self.input_conv2.out_planes, act_type=self.input_conv2.act_type)
            ## update weights & bias for conv3x3
            conv.conv3x3.weight.data = RK
            conv.conv3x3.bias.data   = RB
            ## update weights & bias for activation
            if self.input_conv2.act_type == 'prelu':
                conv.input_conv2.weight = self.input_conv2.act.weight
            ## update block for backbone
            self.input_conv2 = conv.to(RK.device)

class RepNetwork_V010_BestStruct_teacher_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(RepNetwork_V010_BestStruct_teacher_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
       
        backbone = []
        backbone2 = []
        self.pixelUnShuffle = nn.PixelUnshuffle(3)
        self.head = Conv3X3(inp_planes=self.colors*9, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type)]
        
        self.backbone = nn.Sequential(*backbone)

        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        #conv_bn(in_channels=self.channel_nums, out_channels=self.colors*9, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_type='linear', assign_type=OREPA))
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*9, act_type='linear'))
                                        #Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*9, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(3)

        self.head2 = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)

        for i in range(5):
            backbone2 += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone2 = nn.Sequential(*backbone2)

        self.input_conv2 = Conv3X3(inp_planes=self.colors, out_planes=self.colors*self.scale*self.scale, act_type='linear')
        
        self.upsampler1 = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        
        y0 = self.pixelUnShuffle(x)
        y0 = self.head(y0)
        y = self.backbone(y0) 
        
        #y = torch.cat([y, x], dim=1)
        
        y = self.transition(y)
        y = self.upsampler(y) 
        y = self.head2(y+x)
        y = self.backbone2(y)
        y = self.input_conv2(y)
        y = self.upsampler1(y) 
        return y
    