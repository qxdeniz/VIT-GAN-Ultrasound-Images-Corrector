import torch
import torch.optim as optim
from data_loader import get_dataloaders
from generator import ConditionalGenerator
from discriminator import Discriminator
from vit_quality_assessor import ViTQualityAssessor
from config import Config
import albumentations as A
import torch.nn as nn

def train():
    
    # Инициализация
    gen = ConditionalGenerator().to(Config.DEVICE)
   
    disc = Discriminator().to(Config.DEVICE)

    
    vit = ViTQualityAssessor().to(Config.DEVICE)


    # Оптимизаторы
    opt_gen = optim.Adam(gen.parameters(), lr=Config.LR)
    opt_disc = optim.Adam(disc.parameters(), lr=Config.LR)
    opt_vit = optim.Adam(vit.parameters(), lr=Config.LR)

    # Loss-функции
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_meta = nn.MSELoss()  # Для согласованности с ViT

    # Аугментации
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ], additional_targets={'target': 'image'})

    # Данные
    train_loader, test_loader = get_dataloaders(
        data_root="split_data",
        batch_size=Config.BATCH_SIZE,
        train_transform=train_transform
    )

    for epoch in range(Config.EPOCHS):
        for batch in train_loader:
            if len(batch) == 3: 
                degraded, normal, metadata = batch
                metadata = metadata.to(Config.DEVICE)
            else:
                degraded, normal = batch
                metadata = None
            
            degraded = degraded.to(Config.DEVICE)
            normal = normal.to(Config.DEVICE)

            # 1. Оценка качества ViT
            quality = vit(degraded)  # (batch_size, vit_output_dim)
            
            # 2. Подготовка условий для генератора
            if metadata is not None:
                
                metadata = metadata.float()  

                if len(quality.shape) != len(metadata.shape):
                    quality = quality.view(quality.size(0), -1)  
                    metadata = metadata.view(metadata.size(0), -1)  

                conditions = torch.cat([quality, metadata], dim=1)
            else:
                conditions = quality

            # 3. Обучение дискриминатора
            fake = gen(degraded, conditions)  # Генерация с учетом условий
            disc_real = disc(normal)
            disc_fake = disc(fake.detach())
            loss_disc = (criterion_gan(disc_real, torch.ones_like(disc_real)) + 
                         criterion_gan(disc_fake, torch.zeros_like(disc_fake))) / 2
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # 4. Обучение генератора
            disc_fake = disc(fake)
            loss_gen_gan = criterion_gan(disc_fake, torch.ones_like(disc_fake))
            loss_gen_l1 = criterion_l1(fake, normal)
            
            # Новый loss: согласованность между качеством оригинала и результата
            quality_fake = vit(fake)
            loss_meta = criterion_meta(quality, quality_fake)
            
            # Комбинированный loss
            loss_gen = loss_gen_gan + 10*loss_gen_l1 + 2*loss_meta
            
            opt_gen.zero_grad()
            opt_vit.zero_grad()  # Также обновляем ViT
            loss_gen.backward()
            opt_gen.step()
            opt_vit.step()

        # Валидация
        with torch.no_grad():
            val_loss = 0
            for batch_val in test_loader:
                if len(batch_val) == 3:
                    degraded_val, normal_val, metadata_val = batch_val
                    metadata_val = metadata_val.to(Config.DEVICE)
                else:
                    degraded_val, normal_val = batch_val
                    metadata_val = None

                degraded_val = degraded_val.to(Config.DEVICE)
                normal_val = normal_val.to(Config.DEVICE)

                quality_val = vit(degraded_val)

                if metadata_val is not None:
                    metadata_val = metadata_val.float()
                    if len(quality_val.shape) != len(metadata_val.shape):
                        quality_val = quality_val.view(quality_val.size(0), -1)
                        metadata_val = metadata_val.view(metadata_val.size(0), -1)
                    conditions_val = torch.cat([quality_val, metadata_val], dim=1)
                else:
                    conditions_val = quality_val

                fake_val = gen(degraded_val, conditions_val)
                val_loss += criterion_l1(fake_val, normal_val).item()
            
            print(f"Epoch {epoch}, Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}, Val Loss: {val_loss/len(test_loader):.4f}")

    # Сохранение моделей
    torch.save(gen.state_dict(), "generator.pth")
    torch.save(vit.state_dict(), "vit.pth")

if __name__ == "__main__":
    train()