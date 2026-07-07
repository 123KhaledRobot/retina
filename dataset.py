import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset

from PIL import Image
from preprocessing import pipeline

import os
import random
import cv2

def spatial_augment(img_array):
    if random.random() < 0.5:
        img_array = np.fliplr(img_array).copy()
    if random.random() < 0.5:
        img_array = np.flipud(img_array).copy()
    angle = random.uniform(-20, 20)
    h, w = img_array.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_array = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img_array


class DR(Dataset):
  def __init__(self, csv_file, img_dir, cache_dir="cache",augment=False):
    self.df = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.cache_dir = cache_dir
    self.augment=augment
    os.makedirs(cache_dir, exist_ok=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    filename = self.df.iloc[index]['image']
    label = self.df.iloc[index]['level']

    cache_path = f"{self.cache_dir}/{filename}.npy"

    if os.path.exists(cache_path):
        processed_img = np.load(cache_path)
    else:
        img = Image.open(f"{self.img_dir}/{filename}.jpeg")
        processed_img = pipeline(img)
        np.save(cache_path, processed_img)

    if self.augment:
      processed_img = spatial_augment(processed_img)
    processed_img = np.transpose(processed_img, (2, 0, 1))
    img_tensor = torch.from_numpy(processed_img).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return (img_tensor, label)


# class DR(Dataset):#inheriting features available in the base class Dataset
#   def __init__(self,csv_file,img_dir):
#     #load the csv into df, store the image directory
#     self.df=pd.read_csv(csv_file)
#     self.img_dir=img_dir

#   def __len__(self):
#     #return the # of rows of the dataframe
#     return len(self.df)

#   def __getitem__(self, index):
#     filename=self.df.iloc[index]['image'] #get the filename which is a row in the df based on int value
#     label=self.df.iloc[index]['level']
#     img=Image.open(f"{self.img_dir}/{filename}.jpeg")

#     # img=np.array(img) 
#     """
#     shape is (H,W,3). the last axis is the number of channels. that's how numpy stores it
#     however, pytorch reads as (3,H,W). 
#     """

#     # img=img.astype(np.float32) 

#     processed_img=pipeline(img)
#     processed_img=np.transpose(processed_img,(2,0,1)) #this makes the axes as (3,H,W)
#     img_tensor = torch.from_numpy(processed_img).float() / 255.0

#     """
#     this converts the dtype to float32. by default, it is uint8 which stores integer values
#     for example, 200/255 is .78 which gets rounded to 1. this is not accurate
#     btw, backpropagation requires the gradient values to be in decimals
#     """

#     """
#     we will call preprocessing.py here to implement the cropping, CLAHE,etc
#     """

#     # img=img/255.0 #normalizing values between 0 and 1

#     # img_tensor=torch.Tensor(img)


#     # ImageNet normalization
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     img_tensor = (img_tensor - mean) / std


#     return (img_tensor,label)



