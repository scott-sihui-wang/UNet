import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line

from torchvision import transforms
import math

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data
        self.images = sorted([os.path.join(os.path.join(self.data_dir, 'scans'),f) for f in os.listdir(os.path.join(self.data_dir, 'scans'))])
        self.labels = sorted([os.path.join(os.path.join(self.data_dir, 'labels'),f) for f in os.listdir(os.path.join(self.data_dir, 'labels'))])

    def __getitem__(self, idx):
        # todo

        # load image and mask from index idx of your data
        if self.train == True:
        	image = Image.open(self.images[idx])
        	label = Image.open(self.labels[idx])
        	#print("Train:",self.labels[idx])
        else:
        	image = Image.open(self.images[len(self.images) - 1 - idx])
        	label = Image.open(self.labels[len(self.images) - 1 - idx])
        	#print("Test:",self.labels[len(self.images) - 1 - idx])
        transform_list = []
        transform_list.append(transforms.Grayscale(1))
        transform_list.append(transforms.Resize(self.size))
        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # todo
                # flip image vertically
                image = transforms.functional.vflip(image)
                label = transforms.functional.vflip(label)
            elif augment_mode == 1:
                # todo
                # flip image horizontally
                image = transforms.functional.hflip(image)
                label = transforms.functional.hflip(label)
            elif augment_mode == 2:
                # todo
                # zoom image
                transform_list.append(transforms.Resize(np.random.randint(self.size//4, self.size*4)))
                transform_list.append(transforms.CenterCrop(self.size))
                
            elif augment_mode == 3:
                # todo
                # rotate image
                transform_list.append(transforms.RandomRotation(degrees = (0,180),fill=(0,)))
                
            elif augment_mode == 4:
            	 # gamma correction
            	 image = transforms.functional.adjust_gamma(image,3.0)

        # todo
        # return image and mask in tensors
        transform_list.append(transforms.ToTensor())
        trans = transforms.Compose(transform_list)
        image = trans(image)
        label = trans(label) * 255
        
        return image, label

    def __len__(self):
    	if self.train == True:
    		return math.floor(len(self.images) * self.train_test_split)
    	else:
    		return math.ceil(len(self.images) * (1 - self.train_test_split))

