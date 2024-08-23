from .RepNetworkPlain import RepNetwork_V010_BestStruct_deploy
from .RepNetworkPlain import *
from .RepNetworkPlain_tea import *
from .RepConv_block import RepBlockV5
def init_weights(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    
def get_model(cfg, device, mode='Train'):
    if cfg.model == 'RepNetwork_V012_BestStruct':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5, bias=cfg.bias)
        else: 
            model = RepNetwork_V010_BestStruct_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, bias=cfg.bias)

    ###teacher model    
    elif cfg.model == 'RepNetwork_V012_BestStruct_teacher':
        if mode == 'Train':
            model = RepNetwork_V011_BestStruct_teacher(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors, block_type=RepBlockV5)
        else: 
            model = RepNetwork_V010_BestStruct_teacher_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)

    else: 
        raise NameError('Choose proper model name!!!')
        
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
