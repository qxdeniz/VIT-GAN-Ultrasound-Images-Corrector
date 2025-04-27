import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_quality_metrics(normal_img, degraded_img):
    if len(normal_img.shape) == 3:
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    if len(degraded_img.shape) == 3:
        degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)

    diff = normal_img.astype(float) - degraded_img.astype(float)
    
    return {
        'brightness_diff': np.mean(diff),
        'contrast_diff': np.std(degraded_img) - np.std(normal_img),
        'noise_level': np.std(degraded_img[::10, ::10]), 
        'ssim': compute_ssim(normal_img, degraded_img),
        'is_degraded': 1
    }

def compute_ssim(img1, img2):
   
    return 0.9 

def generate_metadata(data_root="split_data"):

    metadata = []
    
    for mode in ['train', 'test']:
        normal_dir = os.path.join(data_root, mode, "normal")
        degraded_dir = os.path.join(data_root, mode, "degraded")
        
        for img_name in tqdm(os.listdir(degraded_dir), desc=f"Processing {mode}"):
            normal_path = os.path.join(normal_dir, img_name)
            degraded_path = os.path.join(degraded_dir, img_name)
            
            if os.path.exists(normal_path):
                normal_img = cv2.imread(normal_path)
                degraded_img = cv2.imread(degraded_path)
                
                metrics = calculate_quality_metrics(normal_img, degraded_img)
                metrics.update({
                    'filename': img_name,
                    'dataset': mode,
                    'normal_path': normal_path,
                    'degraded_path': degraded_path
                })
                metadata.append(metrics)
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(data_root, "metadata.csv"), index=False)
    return df


generate_metadata()