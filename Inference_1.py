import argparse
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

# ==========================================
#  1. NATIVE PYTORCH UNET ARCHITECTURE
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(F.max_pool2d(x1, 2)))
        u1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        u1_processed = F.relu(self.dec2(u1))
        x3 = u1_processed + x1 
        return self.dec1(x3)

# ==========================================
#  2. SELF-CONTAINED DDIM SAMPLER
# ==========================================
def ddim_sample_loop(model, shape, condition_tensor, device, steps=50):
    """Generates predictions without needing gaussian_diffusion files."""
    model.eval()
    img = th.randn(shape, device=device)
    
    betas = th.linspace(0.0001, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = th.cumprod(alphas, dim=0)

    with th.no_grad():
        for i in reversed(range(0, steps)):
            input_tensor = th.cat((condition_tensor, img), dim=1) if condition_tensor is not None else img

# ==========================================
#  3. CLI ARGUMENT PARSING
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SimpleUNet")
    parser.add_argument('--livs_weights', type=str, required=True, help="Path to the model weights file")
    parser.add_argument('--test_img_dir', type=str, required=True, help="Directory containing test medical images")
    args = parser.parse_args()

