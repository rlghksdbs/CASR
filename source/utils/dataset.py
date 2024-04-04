import os
import numpy as np
from PIL import Image
from typing import List, Callable, Tuple
from sklearn import naive_bayes


import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from source.utils import util_image as util


# Transforms
class Compose:
    def __init__(self,
                 transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self,
                 lr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            lr = transform(lr)

        return lr


class ToTensor:
    def __init__(self,
                 rgb_range: int = 1):
        self.rgb_range = rgb_range

    def __call__(self,
                 lr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_range != 1:
            lr = F.pil_to_tensor(lr).float()
        else:
            lr = F.to_tensor(np.array(lr))

        return lr
    

class CenterCrop:
    def __init__(self,
                 crop_size: int,
                 scale: int):
        self.h, self.w = crop_size
        self.scale = scale

    def __call__(self,
                 lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        lr = F.center_crop(lr, (int(self.h/self.scale), int(self.w/self.scale)))

        return lr


# Dataset
class SRDataset(data.Dataset):
    def __init__(self, lr_images_dir, transform=None, n_channels=3):
        self.lr_images_dir = lr_images_dir
        self.lr_images = sorted(os.listdir(lr_images_dir))
        self.transform = transform
        self.n_channels = n_channels

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.lr_images_dir, self.lr_images[index])
       
        img_L = util.imread_uint(img_path, n_channels=self.n_channels)
        img_L = util.uint2tensor3(img_L)


        if self.transform:
            img_L = self.transform(img_L)

        return img_L, self.lr_images[index]

class Eval_Dataset(data.Dataset):
    def __init__(self, lr_images_dir, hr_images_dir, transform=None, n_channels=3, scale=4, down_size=2):
        self.lr_images_dir = lr_images_dir
        self.lr_images = sorted(os.listdir(lr_images_dir))
        
        self.hr_images_dir = hr_images_dir
        self.hr_images = sorted(os.listdir(hr_images_dir))

        self.transform = transform
        self.n_channels = n_channels
        self.down_size = down_size
        self.scale = scale

    def __len__(self):
        if len(self.lr_images) == len(self.hr_images):
            return len(self.lr_images)

    def __getitem__(self, index):
        lr_img_path = os.path.join(self.lr_images_dir, self.lr_images[index])
        hr_img_path = os.path.join(self.hr_images_dir, self.hr_images[index])

        img_L = util.imread_unit_avif(lr_img_path, n_channels=self.n_channels)
        img_L = util.uint2tensor3(img_L)

        img_H = util.imread_uint(hr_img_path, n_channels=self.n_channels)
        img_H = util.uint2tensor3(img_H)

        _, lr_h, lr_w = img_L.shape
        if lr_h % self.down_size != 0:
            for j in range(1, self.down_size):
                if (lr_h - j) % self.down_size == 0:
                    lr_h = lr_h - j
        elif  lr_h*self.scale > img_H.shape[1]:
            lr_h = lr_h - self.down_size
            
        if lr_w % self.down_size != 0:
            for j in range(1, self.down_size):
                if (lr_w - j) % self.down_size == 0:
                    lr_w = lr_w - j
        elif  lr_w*self.scale > img_H.shape[2]:
            lr_w = lr_w - self.down_size
        
        img_L_new = img_L[:, 0:lr_h,       0:lr_w]
        img_H_new = img_H[:, 0:lr_h*self.scale, 0:lr_w*self.scale]
    
        if self.transform:
            img_L_new = self.transform(img_L_new)
            img_H_new = self.transform(img_H_new)

        return img_L_new, self.lr_images[index], img_H_new, self.hr_images[index]