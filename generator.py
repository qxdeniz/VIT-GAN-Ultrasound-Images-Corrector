import torch.nn as nn
from config import Config
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConditionalGenerator(nn.Module):
    def __init__(self, condition_dim=10):  # Изменяем размерность на 10 (5 от ViT + 5 от метаданных)
        super().__init__()
        
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, 256),  #10 признаков
            nn.ReLU(),
            nn.Linear(256, Config.GEN_FILTERS*8),
            nn.ReLU()
        )
        
        # Энкодер
        self.down1 = DoubleConv(1, Config.GEN_FILTERS)
        self.down2 = DoubleConv(Config.GEN_FILTERS, Config.GEN_FILTERS*2)
        self.down3 = DoubleConv(Config.GEN_FILTERS*2, Config.GEN_FILTERS*4)
        
        self.bottleneck = DoubleConv(Config.GEN_FILTERS*4, Config.GEN_FILTERS*8)
        
        # Декодер
        self.up1 = nn.ConvTranspose2d(Config.GEN_FILTERS*8, Config.GEN_FILTERS*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(Config.GEN_FILTERS*4, Config.GEN_FILTERS*2, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(Config.GEN_FILTERS*2, Config.GEN_FILTERS, 2, stride=2)
        
        self.final_conv = nn.Conv2d(Config.GEN_FILTERS, 1, 1)
        
        self.adain1 = nn.Linear(Config.GEN_FILTERS*8, Config.GEN_FILTERS*4)
        self.adain2 = nn.Linear(Config.GEN_FILTERS*8, Config.GEN_FILTERS*2)
        self.adain3 = nn.Linear(Config.GEN_FILTERS*8, Config.GEN_FILTERS)

    def forward(self, x, conditions=None):
        if conditions is not None:
            style = self.condition_proj(conditions)  # (B, GEN_FILTERS*8)
        
        # Энкодер
        x1 = self.down1(x)
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        
        x = self.bottleneck(nn.MaxPool2d(2)(x3))
        if conditions is not None:
            x = self.apply_adain(x, style)
        
        x = self.up1(x)
        if conditions is not None:
            x = self.apply_adain(x, self.adain1(style))
        x = x + x3
        
        x = self.up2(x)
        if conditions is not None:
            x = self.apply_adain(x, self.adain2(style))
        x = x + x2
        
        x = self.up3(x)
        if conditions is not None:
            x = self.apply_adain(x, self.adain3(style))
        x = x + x1
        
        return torch.sigmoid(self.final_conv(x))
    
    def apply_adain(self, x, style):
        bs, ch = x.size(0), x.size(1)
        style = style.view(bs, ch, 1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return style * (x - mean) / (std + 1e-7) + style