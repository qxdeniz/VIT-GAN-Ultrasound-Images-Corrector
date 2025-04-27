import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import pandas as pd
import albumentations as A


class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_pairs = []
        self.metadata = []

        metadata_path = os.path.join(root_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            self.df_metadata = pd.read_csv(metadata_path)
        else:
            self.df_metadata = None
        
        normal_dir = os.path.join(self.root_dir, "normal")
        degraded_dir = os.path.join(self.root_dir, "degraded")
        
        for img_name in os.listdir(degraded_dir):
            degraded_path = os.path.join(degraded_dir, img_name)
            normal_path = os.path.join(normal_dir, img_name)
            
            if os.path.exists(normal_path):
                self.image_pairs.append((degraded_path, normal_path))

                if self.df_metadata is not None:
                    meta = self.df_metadata[self.df_metadata['filename'] == img_name].to_dict('records')
                    self.metadata.append(meta[0] if meta else {})

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        degraded_path, normal_path = self.image_pairs[idx]
        
        degraded_img = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        normal_img = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        if self.transform:
            transformed = self.transform(image=degraded_img, target=normal_img)
            degraded_img, normal_img = transformed['image'], transformed['target']
        
        degraded_img = np.expand_dims(degraded_img, axis=0)  # (1, H, W)
        normal_img = np.expand_dims(normal_img, axis=0)

        if self.metadata:
            # Extract only numerical columns (first 5 columns: brightness_diff, contrast_diff, noise_level, ssim, is_degraded)
            metadata_values = [
                self.metadata[idx]['brightness_diff'],
                self.metadata[idx]['contrast_diff'],
                self.metadata[idx]['noise_level'],
                self.metadata[idx]['ssim'],
                self.metadata[idx]['is_degraded']
            ]
            return (
                torch.tensor(degraded_img),
                torch.tensor(normal_img),
                torch.tensor(metadata_values, dtype=torch.float32)
            )
        
        return torch.tensor(degraded_img), torch.tensor(normal_img)

def get_dataloaders(data_root="split_data", batch_size=Config.BATCH_SIZE, train_transform=None):
    # Create a basic resize transform for test data
    test_transform = A.Compose([
        A.Resize(height=256, width=256)  # Same size as training data
    ], additional_targets={'target': 'image'})

    train_dataset = UltrasoundDataset(data_root, mode='train', transform=train_transform)
    test_dataset = UltrasoundDataset(data_root, mode='test', transform=test_transform)  # Add transform here
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader