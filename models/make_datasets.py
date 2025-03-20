
# 数据处理

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# 加载数据集    随机transform变换导致不一致  需要联合变换
class MyGODataset(Dataset):
    def __init__(self, hr_dir,lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform

        self.img_names = os.listdir(self.hr_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name)

        hr=Image.open(hr_path)
        lr=Image.open(lr_path)

        hr=self.transform(hr)
        lr=self.transform(lr)

        return hr, lr


# 加载数据集    增加了翻转/旋转增强
class MyGODataset_eh(Dataset):
    def __init__(self, hr_dir,lr_dir,flip=False,Rotate=False, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.flip=flip
        self.Rotate=Rotate

        self.img_names = os.listdir(self.hr_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name)

        hr=Image.open(hr_path)
        lr=Image.open(lr_path)

        hr=self.transform(hr)
        lr=self.transform(lr)
        if self.flip == True:
            hr = torch.flip(hr, dims=[2])
            lr = torch.flip(lr, dims=[2])
        if self.Rotate == True:
            hr = hr.transpose(1, 2)
            lr = lr.transpose(1, 2)

        return hr, lr


# 加载数据集    分别加载高分辨率图像与低分辨率图像  --------------- 修改这里
class MyGODataset_valid(Dataset):
    def __init__(self, hr_dir,lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.img_names_hr = os.listdir(self.hr_dir)
        # self.img_names_lr = os.listdir(self.lr_dir)

    def __len__(self):
        return len(self.img_names_hr)

    def __getitem__(self, idx):
        img_name_hr = self.img_names_hr[idx]
        # img_name_lr = self.img_names_lr[idx]
        base, ext = os.path.splitext(img_name_hr)           # 分离基本名和扩展名
        img_name_lr = f"{base}x4{ext}"                      # 映射到低分辨率图像
        hr_path = os.path.join(self.hr_dir, img_name_hr)    
        lr_path = os.path.join(self.lr_dir, img_name_lr)    

        hr=Image.open(hr_path)
        lr=Image.open(lr_path)
        hr=self.transform(hr)
        lr=self.transform(lr)
        return hr, lr


# 成对 HR-LR 图像的加载
class MyGODataset_ls(Dataset):
    def __init__(self, hr_dir,lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.folder_names_hr = os.listdir(self.hr_dir)
        self.folder_names_lr = os.listdir(self.lr_dir)

        self.img_list_hr=[]
        self.img_list_lr=[]
        for folder in self.folder_names_hr:
            folder_path_hr = os.path.join(hr_dir, folder)
            folder_path_lr = os.path.join(lr_dir, folder)
            for img_name in os.listdir(folder_path_hr):
                img_name_hr = os.path.join(folder_path_hr, img_name)
                img_name_lr = os.path.join(folder_path_lr, img_name)

                self.img_list_hr.append(img_name_hr)
                self.img_list_lr.append(img_name_lr)

    def __len__(self):
        return len(self.img_list_hr)

    def __getitem__(self, idx):
        img_name_hr = self.img_list_hr[idx]
        img_name_lr = self.img_list_lr[idx]

        hr=Image.open(img_name_hr)
        lr=Image.open(img_name_lr)

        hr=self.transform(hr)
        lr=self.transform(lr)

        return hr, lr
