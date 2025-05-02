import torch

class Config:
    IMAGE_SIZE = 256  # Make sure this matches the size used during training
    BATCH_SIZE = 8
    
    # ViT
    VIT_PATCH_SIZE = 16
    VIT_DIM = 128
    VIT_DEPTH = 4
    
    # Генератор
    GEN_FILTERS = 64  # Base number of filters for generator
    
    # Обучение
    LR = 0.0002
    EPOCHS = 55
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()