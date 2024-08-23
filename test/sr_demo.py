import os
import torch
import pathlib
import logging
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import source.models
import source.utils.dataset as dd

from source.utils import util_logger
from source.utils import util_image as util
from source.utils.model_summary import get_model_flops
from source.models.get_model import get_model
import yaml

def main(args):
    """
    SETUP DIRS
    """
    pathlib.Path(os.path.join(args.save_dir, args.submission_id, "results")).mkdir(parents=True, exist_ok=True)
    
    """
    SETUP LOGGER
    """
    util_logger.logger_info("AIS2024-RTSR", log_path=os.path.join(args.save_dir, args.submission_id, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("AIS2024-RTSR")
    
    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:{}'.format(0))
    
    """
    LOAD MODEL
    """
    if not args.bicubic:
        if args.config:
            opt = vars(args)
            yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
            opt.update(yaml_args)
        model = get_model(args, device, mode='Deploy')
        #model = models.__dict__[args.model_name]()
        
        if args.checkpoint is not None:
            model_path = os.path.join(args.checkpoint)
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        
        # number of parameters
        number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('Params number: {}'.format(number_parameters))
        print(model)
    
    """
    SETUP DATALOADER
    """
    dataset = dd.SRDataset(lr_images_dir=args.lr_dir, n_channels=3, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    """
    TESTING
    """ 
    with torch.no_grad():          
        for img_L, img_path in tqdm(dataloader):
            img_name, ext = os.path.splitext(img_path[0])

            # load LR image
            img_L = img_L.to(device, non_blocking=True)
            
            # forward pass
            if args.bicubic:
                img_E = F.interpolate(img_L, scale_factor=args.scale, mode="bicubic", align_corners=False)
            else:
                if args.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        img_E = model(img_L)
                else:
                    img_E = model(img_L)
            
            if args.save_sr:    
                # postprocess
                img_E = util.tensor2uint(img_E)
                
                # save model output
                util.imsave(img_E, os.path.join(os.path.join(args.save_dir, args.submission_id, "results", img_name + ".png")))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--submission-id", type=str, default='0002')
    #parser.add_argument("--model-name", type=str, choices=["swin2sr", "imdn", "rfdn"], default='RepConv')
    parser.add_argument("--checkpoint", type=str, default='./WEIGHT_RESULT/Candidate/m4c64/PlainRepConv_BlockV2_x3_p384_m4_c64_relu_l2_adam_lr0.0001_e200_t2023-0321-1800_combined3_psnr_28_74/models/model_x3_best_submission_deploy.pt')
    parser.add_argument("--save-dir", type=str, default="./WEIGHT_RESULT/Candidate/m4c64/PlainRepConv_BlockV2_x3_p384_m4_c64_relu_l2_adam_lr0.0001_e200_t2023-0321-1800_combined3_psnr_28_74/")
    parser.add_argument('--config', type=str, default='./configs/x4.yml', help = 'pre-config file for training')
    
    # specify dirs
    parser.add_argument("--lr-dir", type=str, default='../datasets/RTSR/val_lr')
    parser.add_argument("--save-sr", action="store_true", default=True)
    
    # specify test case
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    #parser.add_argument("--crop-size", type=int, nargs="+", default=[1080, 2040])
    # parser.add_argument("--crop-size", type=int, nargs="+", default=[720, 1280])
    parser.add_argument("--crop-size", type=int, nargs="+", default=[540, 960])
    parser.add_argument("--bicubic", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()
        
    main(args)