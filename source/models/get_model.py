from .plainsr import PlainSR, PlainSR2
from .plainRepConv import PlainRepConv, PlainRepConv_st01, PlainRepConv_BlockV2, PlainRepConv_All, PlainRepConvClip, PlainRepConv_deploy, PlainRepConv_BlockV2_deploy
from .imdn_baseline import IMDN
from .fsrcnn import FSRCNN
from .rtsrn import RTSRN
from .downRepConv import DownRepConv_Block, DownRepConv_Block_deploy
from .downRepConv_v2 import DownRepConv_Block_v2, DownRepConv_Block_v2_deploy, DownRepConv_Block_v2_backbone_res, DownRepConv_Block_v2_backbone_res_deploy

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
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
