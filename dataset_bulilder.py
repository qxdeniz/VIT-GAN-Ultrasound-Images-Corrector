import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = "data"
normal_dir = os.path.join(data_dir, "normal")
degraded_dir = os.path.join(data_dir, "degrades")

output_dir = "split_data"
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(os.path.join(train_dir, "normal"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "degraded"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "normal"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "degraded"), exist_ok=True)

pairs = []
for normal_img in os.listdir(normal_dir):
    degraded_img = normal_img  
    normal_path = os.path.join(normal_dir, normal_img)
    degraded_path = os.path.join(degraded_dir, degraded_img)
    
    if os.path.exists(degraded_path):
        pairs.append((normal_path, degraded_path))

train_pairs, test_pairs = train_test_split(
    pairs, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

def copy_files(pairs, normal_dst, degraded_dst):
    for normal_src, degraded_src in pairs:
        shutil.copy(normal_src, os.path.join(normal_dst, os.path.basename(normal_src)))
        shutil.copy(degraded_src, os.path.join(degraded_dst, os.path.basename(degraded_src)))

copy_files(train_pairs, 
           os.path.join(train_dir, "normal"),
           os.path.join(train_dir, "degraded"))

copy_files(test_pairs,
           os.path.join(test_dir, "normal"),
           os.path.join(test_dir, "degraded"))

print(f"Train: {len(train_pairs)} пар")
print(f"Test: {len(test_pairs)} пар")