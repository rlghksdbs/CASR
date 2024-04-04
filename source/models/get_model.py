from .plainsr import PlainSR, PlainSR2
from .plainRepConv import PlainRepConv, PlainRepConv_st01, PlainRepConv_BlockV2, PlainRepConv_All, PlainRepConvClip, PlainRepConv_deploy, PlainRepConv_BlockV2_deploy
from .imdn_baseline import IMDN
from .fsrcnn import FSRCNN
from .rtsrn import RTSRN
from .downRepConv import DownRepConv_Block, DownRepConv_Block_deploy
from .downRepConv_v2 import DownRepConv_Block_v2, DownRepConv_Block_v2_deploy, DownRepConv_Block_v2_backbone_res, DownRepConv_Block_v2_backbone_res_deploy
from .RepNetworkPlain import RepNetwork_V001, RepNetwork_V002, RepNetwork_V004_BestStruct, RepNetwork_V005_BestStruct, RepNetwork_V006_BestStruct, RepNetwork_V007_BestStruct, RepNetwork_V010_BestStruct, RepNetwork_V010_BestStruct_deploy
from .RepNetworkPlain import *
from .RepNetworkPlain_tea import *
from .RepConv_block import RepBlock, RepBlockV2, RepBlockV3, RepBlockV4, RepBlockV5, RepBlockV6, RepBlockV7, RepBlockV8, RepBlockV9, RepBlockV10
from .bicubic_plus_plus import Bicubic_plus_plus
from .AsConvSR import AsConvSR
from .RepNetworkPlain_fidelity import RepNetwork_V011_fidelity_BestStruct, RepNetwork_V011_fidelity_BestStruct_V2, RepNetwork_V011_fidelity_BestStruct_V3, RepNetwork_V010_BestStruct_fidelity_deploy
def init_weights(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    
def get_model(cfg, device, mode='Train'):
    if cfg.model == 'plainsr':
        model = PlainSR(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'plainsr2':
        model = PlainSR2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv':
        if mode == 'Train':
            model = PlainRepConv(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConvClip':
        model = PlainRepConvClip(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_st01':
        model = PlainRepConv_st01(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_All':
        model = PlainRepConv_All(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)    
    elif cfg.model == 'PlainRepConv_BlockV2':
        if mode == 'Train':
            model = PlainRepConv_BlockV2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_BlockV2_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'IMDN':
        model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=cfg.scale, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05)
    elif cfg.model == "FSRCNN":
        model = FSRCNN(colors=3, upscale_factor=cfg.scale)
    elif cfg.model == "RTSRN":
        model = RTSRN(num_channels=3, num_feats=64, num_blocks=5, upscale=cfg.scale)

    elif cfg.model == 'DownRepConv':
        if mode == 'Train':
            model = DownRepConv_Block(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = DownRepConv_Block_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
   
    elif cfg.model == 'DownRepConv_v2':
        if mode == 'Train':
            model = DownRepConv_Block_v2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = DownRepConv_Block_v2_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'DownRepConv_v2_backbone_res':
        if mode == 'Train':
            model = DownRepConv_Block_v2_backbone_res(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = DownRepConv_Block_v2_backbone_res_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V001':
       model = RepNetwork_V001(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V002':
       model = RepNetwork_V002(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V004_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V004_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, deploy=False)
        else: 
            model = RepNetwork_V004_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, deploy=True)
    elif cfg.model == 'RepNetwork_V005_BestStruct':
       model = RepNetwork_V005_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V006_BestStruct':
       model = RepNetwork_V006_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V007_BestStruct':
       model = RepNetwork_V007_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V010_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V010_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V011_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV4)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V012_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5, bias=cfg.bias)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, bias=cfg.bias)
    elif cfg.model == 'RepNetwork_V013_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV6)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V014_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V014_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV2)
        else: 
            model = RepNetwork_V014_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'RepNetwork_V015_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV7)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)  
    elif cfg.model == 'RepNetwork_V016_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV8)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)   
    elif cfg.model == 'RepNetwork_V005_TypeA':
        if mode == 'Train':
            model = RepNetwork_V005_TypeA(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5)
        else: 
            model = RepNetwork_V005_TypeA_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)   
    elif cfg.model == 'RepNetwork_V030_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV9)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)   
    elif cfg.model == 'RepNetwork_V031_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV10)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors) 
    elif cfg.model == 'RepNetwork_V040_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V111_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5)
        else: 
            model = RepNetwork_V111_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors) 
    ###teacher model    
    elif cfg.model == 'RepNetwork_V012_BestStruct_teacher':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct_teacher(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5)
        else: 
            model = RepNetwork_V010_BestStruct_teacher_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)

    elif cfg.model == 'RepNetwork_V012_BestStruct_fidelity':
        if mode == 'Train':
            model = RepNetwork_V011_fidelity_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5, bias=cfg.bias)
        else: 
            model = RepNetwork_V010_BestStruct_fidelity_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, bias=cfg.bias)
    elif cfg.model == 'RepNetwork_V012_BestStruct_fidelity_V2':
        if mode == 'Train':
            model = RepNetwork_V011_fidelity_BestStruct_V2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5, bias=cfg.bias)
        else: 
            model = RepNetwork_V010_BestStruct_fidelity_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, bias=cfg.bias)
    elif cfg.model == 'RepNetwork_V012_BestStruct_fidelity_V3':
        if mode == 'Train':
            model = RepNetwork_V011_fidelity_BestStruct_V3(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5, bias=cfg.bias)
        else: 
            model = RepNetwork_V010_BestStruct_fidelity_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, bias=cfg.bias)
 
    ###sota model
    elif cfg.model == 'bicubic_plus_plus':
        model = Bicubic_plus_plus(sr_rate=cfg.scale)

    elif cfg.model == 'asconv':
        model = AsConvSR(scale_factor=cfg.scale, device=device)
        #model = AsConvSR(scale_factor=cfg.scale)
    else: 
        raise NameError('Choose proper model name!!!')
        
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
