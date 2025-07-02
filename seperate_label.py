
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import glob
import cv2
from PIL import Image
import copy

# train
dataset_folder = '/mnt/nvme1/suyuejiao/data/data/'
label_folder = dataset_folder + 'train/label/'

train_left_obj_inx_folder = dataset_folder + 'train/lbl_obj_left/'
train_right_obj_inx_folder = dataset_folder + 'train/lbl_obj_right/'
train_two_obj_inx_folder= dataset_folder + 'train/lbl_obj_two/'

os.makedirs(train_left_obj_inx_folder, exist_ok=True)
os.makedirs(train_right_obj_inx_folder, exist_ok=True)
os.makedirs(train_two_obj_inx_folder, exist_ok=True)

# save the left, right, two obj seperately
for file in tqdm(glob.glob(label_folder+'*.png')):
        filename = file.split('/')[-1].split('.')[0]
        obj = np.array(Image.open(file))
        obj_left = np.zeros(obj.shape)
        obj_right = np.zeros(obj.shape)
        obj_two = np.zeros(obj.shape)

        obj_two_mask = obj==5
        obj_two[obj_two_mask] = 1
        obj_two = Image.fromarray(np.uint8(obj_two))
        obj_two.save(train_two_obj_inx_folder+filename+'.png')

        obj_left_mask = obj==3
        obj_left[obj_left_mask] = 1
        obj_left[obj_two_mask] = 1
        obj_left = Image.fromarray(np.uint8(obj_left))
        obj_left.save(train_left_obj_inx_folder+filename+'.png')

        obj_right_mask = obj==4
        obj_right[obj_right_mask] = 1
        obj_right[obj_two_mask] = 1
        obj_right = Image.fromarray(np.uint8(obj_right))
        obj_right.save(train_right_obj_inx_folder+filename+'.png')

        

# -----------------------
# check the intersection of left, right, and two object

for file in tqdm(glob.glob(train_left_obj_inx_folder+'*.png')):
    filename = file.split('/')[-1].split('.')[0]

    obj_two2 = np.array(Image.open(train_two_obj_inx_folder+filename+'.png'))
    if obj_two2.sum()!=0:


        obj_left = np.array(Image.open(file))
        right_obj = np.array(Image.open(train_right_obj_inx_folder+filename+'.png'))

        obj_two1 = np.zeros(obj_left.shape)
        two = obj_left+right_obj
        assert max(np.unique(two))<=2
        mask = two>1
        obj_two1[mask] = 1

        assert ((obj_two1-obj_two2).sum())==0.0


    