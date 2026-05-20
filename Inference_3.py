import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from datetime import datetime

# ==========================================
# 1. ARCHITECTURE
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
# 2. HELPER UTILITIES
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
def process_numpy_volume(vol_data, model, device):
    """Processes a 3D numpy array in (Slice, H, W) format."""
    itk_temp = sitk.GetImageFromArray(vol_data.astype(np.float32))
    itk_temp.SetSpacing([1.0, 1.0, 1.0]) 
    
    img_iso = resample_volume(itk_temp)
    vol_iso = sitk.GetArrayFromImage(img_iso)
    pred_vol = np.zeros_like(vol_iso, dtype=np.uint8)
    
    model.eval()
    for s in range(vol_iso.shape[0]):
        indices = [max(0, s-1), s, min(vol_iso.shape[0]-1, s+1)]
        stack = np.stack([normalize_ct(vol_iso[i]) for i in indices], axis=0)
        
        input_t = th.from_numpy(stack).float().to(device).unsqueeze(0)
        input_t = F.interpolate(input_t, size=(256, 256), mode="bilinear")
        
        noise_ch = th.zeros((1, 1, 256, 256), device=device)
        input_t = th.cat([input_t, noise_ch], dim=1)
        
        output = model(input_t)
        prob_map = th.sigmoid(output)
        
        pred_slice = F.interpolate(prob_map, size=vol_iso.shape[1:], mode="bilinear")
        pred_vol[s] = (pred_slice > 0.5).cpu().numpy().squeeze().astype(np.uint8)
    
    return post_process_clean(pred_vol)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def run_inference(weights_path, input_dir, output_dir):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    model = SimpleUNet(in_channels=4, out_channels=1).to(device)
    checkpoint = th.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    print(f"[{datetime.now():%H:%M}] Model loaded on {device}")

    # 2. Iterate through files in directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    print(f"Found {len(files)} files in {input_dir}")

    for file_name in sorted(files):
        input_path = os.path.join(input_dir, file_name)
        print(f"[{datetime.now():%H:%M}] Processing: {file_name}")
        
        try:
            # Load with Nibabel to handle transposition easily
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # UNPACK/TRANSPOSE: Ensure it's (Slices, H, W)
            # If your data is (H, W, Slices), use (2, 0, 1)
            vol_ready = np.transpose(data, (2, 0, 1))
            
            # Run Inference
            mask = process_numpy_volume(vol_ready, model, device)
            
            # Save Output (Transpose back to original Nifti orientation)
            mask_to_save = np.transpose(mask, (1, 2, 0))
            out_img = nib.Nifti1Image(mask_to_save.astype(np.uint8), img.affine)
            
            out_path = os.path.join(output_dir, f"pred_{file_name}")
            nib.save(out_img, out_path)
            
        except Exception as e:
            print(f"Error on {file_name}: {e}")

if __name__ == "__main__":
    W_PATH = "/home/fader/GATSegDiff/Dockerfile/GATSegDiff/model/LiVS_reannotated_subset_F1_model160000.pt"
    IN_DIR = "/home/fader/CT_Nifti"
    OUT_DIR = "/home/fader/predictions"
    
    run_inference(W_PATH, IN_DIR, OUT_DIR)

