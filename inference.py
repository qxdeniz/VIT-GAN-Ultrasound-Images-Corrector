import os
import cv2
import numpy as np
import torch
from generator import ConditionalGenerator
from vit_quality_assessor import ViTQualityAssessor
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = ConditionalGenerator(condition_dim=10).to(device).eval() 
vit = ViTQualityAssessor().to(device).eval()

gen.load_state_dict(torch.load("generator.pth", map_location=device))
vit.load_state_dict(torch.load("vit.pth", map_location=device))

def enhance_ultrasound(image_path):
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        
    image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        quality = vit(image_tensor)
        
        
        metadata = torch.zeros(1, 5, device=device)  
        conditions = torch.cat([quality.view(1, -1), metadata], dim=1)  
    
        enhanced = gen(image_tensor, conditions)
        enhanced = enhanced.squeeze().cpu().numpy() * 255
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced, quality

def main(test_image):
    try:
        enhanced_img, quality_report = enhance_ultrasound(test_image)
      
        output_path = "enhanced.png"
        cv2.imwrite(output_path, enhanced_img)
        
        quality_metrics = ["brightness", "contrast", "noise", "artifacts", "is_ok"]
        quality_dict = {k: v.item() for k, v in zip(quality_metrics, quality_report.squeeze())}
        
        print("Оценка изображения:")
        for metric, value in quality_dict.items():
            print(f"{metric}: {value:.4f}")

        print("Метрики для улучшенного изображения:")
        res_image = cv2.imread("enhanced.png", cv2.IMREAD_GRAYSCALE)
        image_tensor = torch.tensor(res_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)
    
        with torch.no_grad():
            changed_quality = vit(image_tensor)
            changed_quality_dict = {k: v.item() for k, v in zip(quality_metrics, changed_quality.squeeze())}
            for metric, value in changed_quality_dict.items():
                print(f"{metric}: {value:.4f}")

            
    except Exception as e:
        print(str(e))


main("test.png")