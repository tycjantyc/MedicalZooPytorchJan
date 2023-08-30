import os
import torch
import numpy as np
from torch.utils.data import Dataset
from lib.utils.covid_utils  import  read_txt
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from os.path import join
import SimpleITK as sitk
from lib.medloaders import medical_image_process as img_loader

import nibabel as nib




class Pioro3DDataset(Dataset):
    def __init__(self,mode, split):
        
        self.path_files = "drive/MyDrive/Dane3DHeartCT/images"
        self.path_maski = "drive/MyDrive/Dane3DHeartCT/labels"
        self.full_volume = None
        self.transform = None
        
        self.img_list = []

        

        _,_,files = next(os.walk(self.path_files))
        _,_,maski = next(os.walk(self.path_maski))
        l = len(files)

        if mode == "train":
            for index, (f, m) in enumerate(zip(files, maski)):
                if index < l * split:
                    cls_list = [join(self.path_files, f), join(self.path_maski, m)]
                    self.img_list.append(cls_list)
                else:
                    pass

        if mode == "test":
            for index, (f, m) in enumerate(zip(files, maski)):
                if index >= l * split:
                    cls_list = [join(self.path_files, f), join(self.path_maski, m)]
                    self.img_list.append(cls_list)
                else:
                    pass
        self.affine = None         
        # img_loader.load_affine_matrix(self.img_list[0])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, mask_path = self.img_list[idx]
        
        image, mask = nib.load(img_path), nib.load(mask_path)
        image, mask = np.array(image.dataobj), np.array(mask.dataobj)

        dim = image.shape[2]
        print(mask.shape[2])
        if(dim > 177):
            image = image[:,:,:177]
            mask  =  mask[:,:,:177]
        print(image.shape)
        print(mask.shape)
        if(mask.shape[2] == 363):
          mask = mask[:, :, :177]
          print("got it")
        
        image, mask = image.astype(np.float32), mask.astype(np.float32)
        

        return torch.FloatTensor(image), torch.FloatTensor(mask)
        return image, mask
