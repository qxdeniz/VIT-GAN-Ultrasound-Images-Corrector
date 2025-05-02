import torch
import torch.nn as nn
from einops import rearrange
from config import Config

class ViTQualityAssessor(nn.Module):
    def __init__(self):
        super().__init__()
        patch_size = Config.VIT_PATCH_SIZE
        self.patch_embed = nn.Conv2d(1, Config.VIT_DIM, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.VIT_DIM,
            nhead=4,
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.VIT_DEPTH
        )
        self.head = nn.Sequential(
            nn.Linear(Config.VIT_DIM, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.transformer(x)
        x = x.mean(dim=1)  
        return self.head(x)