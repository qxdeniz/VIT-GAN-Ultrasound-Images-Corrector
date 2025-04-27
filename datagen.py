import os
import cv2
import numpy as np

def degrade_image(image):
    # Добавление шума
    noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
    degraded = image + noise
    
   
    alpha = np.random.uniform(0.7, 1.3)  # Контраст
    beta = np.random.uniform(-0.2, 0.2)  # Яркость
    degraded = cv2.convertScaleAbs(degraded, alpha=alpha, beta=beta)
    
    return np.clip(degraded, 0, 1)


input_dir = "normal"
output_dir = "degrades"


os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        image = cv2.imread(input_path).astype(np.float32) / 255.0  
        
        degraded_image = degrade_image(image)
        degraded_image = (degraded_image * 255).astype(np.uint8)  
        cv2.imwrite(output_path, degraded_image)

print("Обработка завершена!")