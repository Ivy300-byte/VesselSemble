import os
import sys
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import argparse
from datetime import datetime

# ==========================================
# 1. SELF-CONTAINED ARCHITECTURE
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, t=None):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(F.max_pool2d(x1, 2)))
        u1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        u1_processed = F.relu(self.dec2(u1))
        x3 = u1_processed + x1 
        return self.dec1(x3)

# ==========================================
# 2. CORE UTILITIES
# ==========================================
def resample_volume(itk_img, target_spacing=[1.0, 1.0, 1.0]):
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize([int(round(sz * sp / t)) for sz, sp, t in zip(itk_img.GetSize(), itk_img.GetSpacing(), target_spacing)])
    resample.SetOutputOrigin(itk_img.GetOrigin())
    resample.SetOutputDirection(itk_img.GetDirection())
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(itk_img)

def normalize_ct(array):
    array = np.clip(array, -100, 400)
    return ((array + 100) / 250.0) - 1.0

def post_process_clean(mask_array, min_size=150):
    mask_itk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    labeled = sitk.ConnectedComponent(mask_itk)
    relabeled = sitk.RelabelComponent(labeled, minimumObjectSize=min_size)
    cleaned_itk = sitk.BinaryThreshold(relabeled, 1, 100000, 1, 0)
    return sitk.GetArrayFromImage(cleaned_itk)

# ==========================================
# 3. INFERENCE ENGINE
# ==========================================
@th.no_grad()
def process_scan(img_path, out_path, model, device):
    img_itk = sitk.ReadImage(img_path)
    img_iso = resample_volume(img_itk)
    vol_data = sitk.GetArrayFromImage(img_iso)
    pred_vol = np.zeros_like(vol_data, dtype=np.uint8)

    model.eval()
    for s in range(vol_data.shape[0]):
        indices = [max(0, s-1), s, min(vol_data.shape[0]-1, s+1)]
        stack = np.stack([normalize_ct(vol_data[i]) for i in indices], axis=0)
        
        input_t = th.from_numpy(stack).float().to(device).unsqueeze(0)
        input_t = F.interpolate(input_t, size=(256, 256), mode="bilinear")
        noise_ch = th.zeros((1, 1, 256, 256), device=device)
        input_t = th.cat([input_t, noise_ch], dim=1)

        output = model(input_t)
        prob_map = th.sigmoid(output)
        
        pred_slice = F.interpolate(prob_map, size=vol_data.shape[1:], mode="bilinear")
        pred_vol[s] = (pred_slice > 0.5).cpu().numpy().squeeze().astype(np.uint8)

    cleaned_vol = post_process_clean(pred_vol)
    pred_itk = sitk.GetImageFromArray(cleaned_vol)
    pred_itk.CopyInformation(img_iso)
    
    final_save = sitk.Resample(pred_itk, img_itk, sitk.Transform(), sitk.sitkNearestNeighbor)
    sitk.WriteImage(final_save, out_path)

# ==========================================
# 4. MAIN WITH EXPLICIT DIRS
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Vessel Segmentation Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--input", type=str, required=True, help="Directory containing .nii.gz scans")
    parser.add_argument("--output", type=str, required=True, help="Directory to save predictions")
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Load Model
    model = SimpleUNet(in_channels=4, out_channels=1).to(device)
    checkpoint = th.load(args.weights, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    print(f"[{datetime.now():%H:%M}] Model ready on {device}")

    # Process files
    for file in sorted(os.listdir(args.input)):
        if file.endswith(".nii.gz"):
            input_path = os.path.join(args.input, file)
            output_path = os.path.join(args.output, f"pred_{file}")
            
            print(f"[{datetime.now():%H:%M}] Processing {file}...")
            try:
                process_scan(input_path, output_path, model, device)
            except Exception as e:
                print(f"Error on {file}: {e}")

if __name__ == "__main__":
    main()

