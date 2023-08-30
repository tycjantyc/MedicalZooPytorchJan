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



class PioroCTDataset(Dataset):
    def __init__(self,mode, split):
        
        self.path_files = "../data/MST2018_dcm_maska/00"
        self.path_maski = "../data/MST2018_dcm_maska/MST2018_00_maskc"
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

        img_path = self.img_list[idx][0]
        mask_path = self.img_list[idx][1]

        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)
        image -= image.min()
        image /= image.max()
        image = torch.Tensor(image)
        image = image.reshape(1, 512, 512)

        
        mask = Image.open(mask_path).convert('')
        mask = ImageOps.grayscale(mask)
        
        transform = transforms.Compose([
        transforms.PILToTensor()
        ])
        
        mask = transform(mask)

        mask = mask.type(torch.uint8)
        #mask -= mask.min()
        #mask /= mask.max()
        mask = mask.reshape(1, 512, 512)
        #mask = mask[0] + mask[1] + mask[2]/3

        

        if self.transform:
            image = self.transform(image)

        return image, mask
    