# Baseline Trainer Code for Real-Time Super Resolution
An older version implemented based on [SimpleIR](https://github.com/xindongzhang/SimpleIR).

The following is more advanced version implemented by us.
  - Logger -> Wandb
  - Argument parsing -> use config files at ./configs/

### Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/rlghksdbs/Real-TimeSR
cd Real-TimeSR
pip install -r requirements.txt
```

### Docker Setting
```
docker build --tag ais2024 .
nvidia-docker run --name ais2024 -it --gpus all --ipc=host --pid=host -v /your/data/path/:/dataset -v /your/sorce_code/:/source_code --shm-size=64g ntire2023:latest
pip install -U numpy

##If you use RTX A6000
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Dataset of SR

You can download all dataset about AIS2024 from Web [Link](https://drive.google.com/drive/folders/1mD9bNoZDywvobOk1XrZupYKACF_nKN5t?usp=drive_link)

You can download Div2k HR dataset from Web [Link](https://drive.google.com/drive/folders/1abtVNw4gOAnnwMF0t10Kc9uD1UOLqJHl?usp=drive_link)

You can download Div2k LR bicubic dataset from Web [Link] (https://drive.google.com/drive/folders/1eEcok5mPTUM3Qz8CJrQ5oCOy1UG8dePg?usp=drive_link)

You can download Div2k LR AVIF dataset from Web [Link] (https://drive.google.com/drive/folders/1FDrUBefKLWxfDp88t-KNOUbeS8pMC0tR?usp=drive_link)

You can download benchmark from Web [Link] (https://drive.google.com/drive/folders/1G2VTvz1lHChQQcY-H7-XNd84esWR5gkj?usp=drive_link)

Path of Dataset must be set in ./config/*name_of_yaml*.yaml

### Dataset preparation for AVIF LR images
You can generate LR images with compression noise. (FFMpeg 6.1 Version)
```
## LR path & HR path must be set by manually
python png2avif.py 
```

### Dataset preparation for Noised LR images
You can generate LR images with compression noise.
```
## LR path & HR path must be set by manually
python source/data/prepare_data.py 
```

### Training
You could also try less/larger batch-size, if there are limited/enough hardware resources in your GPU-server.
We use simple yamlfile for various settings during training. 
You can set all Parameters at yaml file ***./config/name_of_model.yaml***
```
cd simple_real_time_super_resolution

## If you set all settings correct
python train.py --config ./config/x2_final/repConv_x2_m4c32_relu_div2k_warmup_lr5e-4_b8_p384_normalize.yml
```
### Testing
You can set all Parameters in ***./config/config_base_test.yaml***

```
## For test your model use sr_demo to check inference time.
python sr_demo.py
```

### Check Result
Validation result image, Test result image, Best weight, Last weight and log files saved in this ***./output/{DATE_of_TODAY}/{Last_folder}*** folder.
Wandb result [WANDB](https://wandb.ai/iilab/ECCV_MAI2020_SR)

### Profilling model inference time
You can check ideal model inference time by pytorch profilling tool. You can set all Parameters in ***./config/config_base.yaml***
```
## If you set all settings correct
python profiller_model.py
```
