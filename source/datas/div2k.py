import os
import glob
import random
import pickle
from PIL import Image
import pillow_avif 
import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time
from utils import ndarray2tensor
import albumentations as A

def crop_patch_albumentation(lr, hr, patch_size, scale, augment=True):
    a_aug = True
    
    aug_train = A.Compose([A.ImageCompression(quality_lower=31, quality_upper=61, p=0.5)])
    
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp), random.randrange(0, lr_h - lp)
    hx, hy = lx * scale, ly * scale
    #print(hx, hy)
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
        lr_patch = aug_train(image=lr_patch)['image']
        # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch

def crop_image(lr, hr, down_scale, scale):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    if lr_h % down_scale != 0:
        for j in range(1, down_scale):
            if (lr_h - j) % down_scale == 0:
                lr_h = lr_h - j
    if lr_w % down_scale != 0:
        for j in range(1, down_scale):
            if (lr_w - j) % down_scale == 0:
                lr_w = lr_w - j
    
    lr_new = lr[0:lr_h,       0:lr_w,      :]
    hr_new = hr[0:lr_h*scale, 0:lr_w*scale,:]
    return lr_new, hr_new
    
def crop_patch(lr, hr, patch_size, scale, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp), random.randrange(0, lr_h - lp)
    hx, hy = lx * scale, ly * scale
    #print(hx, hy)
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
        # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch

class DIV2K(data.Dataset):
    def __init__(
        self, HR_folder, LR_folder, CACHE_folder, 
        train=True, augment=True, scale=2, colors=1, 
        patch_size=96, repeat=168, normalize=True, av1=True, qp_value=31, all_qp=False, a_aug=False, down_scale=3
    ):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment   = augment
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.cache_dir = CACHE_folder
        self.normalize = normalize
        self.av1 = av1
        self.qp_value = qp_value
        self.all_qp = all_qp
        self.a_aug = a_aug
        self.down_scale = down_scale

        ## for raw png images
        self.hr_filenames = []
        self.lr_filenames = []
        
        ## for numpy array data
        self.hr_npy_names = []
        self.lr_npy_names = []
        
        ## store in ram
        self.hr_images = []
        self.lr_images = []

        ## number of qp for train and test
        number_of_qp = 5 ### 31, 39, 47, 55, 63
        
        self.lr_filenames = sorted(glob.glob(self.LR_folder + '/*.avif'))

        for idx, lr_name in enumerate(self.lr_filenames):
            #_name = os.path.basename(lr_name)[0:4] + '.png'
            if self.all_qp:
                _name = os.path.basename(lr_name)[:-13] + '.png'
            else:
                _name = os.path.basename(lr_name)[:-7] + '.png'
            self.hr_filenames.append(os.path.join(self.HR_folder, _name))
            
        assert len(self.hr_filenames) == len(self.lr_filenames)
          
        self.nums_trainset = len(self.hr_filenames)

        LEN = self.nums_trainset
        hr_dir = os.path.join(self.cache_dir, 'hr', 'ycbcr' if self.colors==1 else 'rgb')
        lr_dir = os.path.join(self.cache_dir, 'lr_x{}'.format(self.scale), 'ycbcr' if self.colors==1 else 'rgb')
        
        self.hr_dir = hr_dir
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)
        else:
            for i in range(LEN):
                hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
                hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                self.hr_npy_names.append(hr_npy_name)
            
        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)
        else:
            for i in range(LEN):
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.avif', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)

        ## prepare hr images
        if self.all_qp: 
            if len(glob.glob(os.path.join(hr_dir, "*.npy"))) != (len(self.hr_filenames)/number_of_qp):
                for i in range(LEN):
                    if (i+1) % 50 == 0:
                        print("convert {} hr images to npy data!".format(i+1))
                    hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
                    if self.colors == 1:
                        hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]
                    hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
                    hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                    self.hr_npy_names.append(hr_npy_name)
                    np.save(hr_npy_name, hr_image)
            else:
                print("hr npy datas have already been prepared!, hr: {}".format(len(self.hr_npy_names)))
        else: 
            if len(glob.glob(os.path.join(hr_dir, "*.npy"))) != len(self.hr_filenames):
                for i in range(LEN):
                    if (i+1) % 50 == 0:
                        print("convert {} hr images to npy data!".format(i+1))
                    hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
                    if self.colors == 1:
                        hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]
                    hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
                    hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                    self.hr_npy_names.append(hr_npy_name)
                    np.save(hr_npy_name, hr_image)
            else:
                print("hr npy datas have already been prepared!, hr: {}".format(len(self.hr_npy_names)))
        ## prepare lr images
        if len(glob.glob(os.path.join(lr_dir, "*.npy"))) != len(self.lr_filenames):
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("convert {} lr images to npy data!".format(i+1))
                # lr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB")
                lr_image = Image.open(self.lr_filenames[i])
                if self.colors == 1:
                    lr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1]
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.avif', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)
                np.save(lr_npy_name, lr_image)
        else:
            print("lr npy datas have already been prepared!, lr: {}".format(len(self.lr_npy_names)))

    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        
        if self.all_qp:
            lr = np.load(self.lr_npy_names[idx])
            basename = os.path.basename(self.lr_npy_names[idx])[:-12]
            hr = np.load(os.path.join(self.hr_dir,basename+'.npy'))
        else: 
            hr, lr = np.load(self.hr_npy_names[idx]), np.load(self.lr_npy_names[idx])
        
        if self.train:
            if self.a_aug:
                train_lr_patch, train_hr_patch = crop_patch_albumentation(lr, hr, self.patch_size, self.scale, True)
            else:
                train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)
            #print(self.lr_npy_names[idx], train_lr_patch.size(1), train_lr_patch.size(2), train_hr_patch.size(1), train_hr_patch.size(2) )
            if self.normalize:
                return train_lr_patch / 255., train_hr_patch / 255.
            else: 
                return train_lr_patch, train_hr_patch
                
        if self.normalize:
            lr, hr = crop_image(lr, hr, self.down_scale, self.scale)
            return ndarray2tensor(lr / 255.), ndarray2tensor(hr / 255.)
        else: 
            return ndarray2tensor(lr), ndarray2tensor(hr)

if __name__ == '__main__':
    HR_folder = '/dataset/SR/RLSR/DIV2K/train_HR'
    LR_folder = '/dataset/SR/RLSR/DIV2K/train_LR'
    argment   = True
    div2k = DIV2K(HR_folder, LR_folder, augment=True, scale=2, colors=3, patch_size=96, repeat=168, store_in_ram=True)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        tlr, thr, vlr, vhr = div2k[idx]
        print(tlr.shape, thr.shape, vlr.shape, vhr.shape)
    end = time.time()
    print(end - start)