import h5py
import scipy.io as io
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter
import cv2

def generate_density_map(img_path, gt_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Load .mat file
    gt = io.loadmat(gt_path)
    points = gt['image_info'][0,0][0,0][0] # Standard ShanghaiTech format
    
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if x < w and y < h:
            density_map[y, x] = 1
            
    # Apply Gaussian filter to create smooth density map
    density_map = gaussian_filter(density_map, sigma=15)
    
    return density_map

def process_dataset(root_dir):
    # Process ShanghaiTech Part A and B
    for part in ['part_A', 'part_B']:
        for split in ['train_data', 'test_data']:
            img_dir = os.path.join(root_dir, part, split, 'images')
            gt_dir = os.path.join(root_dir, part, split, 'ground-truth')
            output_dir = os.path.join(root_dir, part, split, 'ground-truth-h5')
            
            os.makedirs(output_dir, exist_ok=True)
            
            img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                gt_path = os.path.join(gt_dir, 'GT_' + img_name.replace('.jpg', '.mat'))
                
                if os.path.exists(gt_path):
                    density = generate_density_map(img_path, gt_path)
                    
                    # Save as H5 for fast loading
                    h5_path = os.path.join(output_dir, img_name.replace('.jpg', '.h5'))
                    with h5py.File(h5_path, 'w') as hf:
                        hf['density'] = density
                    print(f"Processed: {img_name}")

if __name__ == "__main__":
    process_dataset("data/shanghaitech")
