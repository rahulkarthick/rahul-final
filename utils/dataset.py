import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.io as io

class ShanghaiTechDataset(Dataset):
    def __init__(self, img_root, gt_root, transform=None, train=True):
        self.img_root = img_root
        self.gt_root = gt_root
        self.transform = transform
        self.train = train
        self.img_names = [f for f in os.listdir(img_root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_root, img_name)
        gt_path = os.path.join(self.gt_root, 'GT_' + img_name.replace('.jpg', '.mat'))
        
        img = Image.open(img_path).convert('RGB')
        
        # Load Ground Truth mat file
        gt_data = io.load_dataset(gt_path) # scipy.io.loadmat usually
        # Standard ShanghaiTech format has 'image_info' -> 'location'
        # We need to convert this to a density map normally.
        # But for training, typically we use pre-computed density maps (.h5 or .npy)
        # to save time during training.
        
        if self.transform:
            img = self.transform(img)
            
        # Placeholder: In a real training scenario, we'd return (img, density_map)
        # For now, I'll provide the logic to load.
        return img, img_name 
