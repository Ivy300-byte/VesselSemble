import os
import re
import SimpleITK as sitk
import numpy as np

import os
import SimpleITK as sitk
import numpy as np

def ensemble_multi_dir(ct_dir, livs_dir, msd_dir, output_dir, filename, data_ratio=0.75):
    # 1. PATH SETUP (Now simple because filenames match!)
    path_ct = os.path.join(ct_dir, filename)
    path_25d = os.path.join(livs_dir, filename)
    path_3d = os.path.join(msd_dir, filename)
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. LOAD & ALIGN
    # Since they are cropped differently, Resample is MANDATORY
    ref_ct = sitk.ReadImage(path_ct)
    
    def load_and_align(path, reference):
        img = sitk.ReadImage(path)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        return sitk.GetArrayFromImage(resampler.Execute(img))

    arr_25d = load_and_align(path_25d, ref_ct)
    arr_3d = load_and_align(path_3d, ref_ct)

    # 3. CHANNEL MAPPING (MSD Specific)
    vessel_3d = np.maximum(arr_3d[1, ...], arr_3d[2, ...]) if arr_3d.ndim == 4 else arr_3d

    # 4. FUSION & ANATOMICAL GATE
    fused = np.maximum(arr_25d * data_ratio, vessel_3d * (1 - data_ratio))
    liver_mask = (arr_25d > 0.25).astype(np.float32)
    final_vessels = fused * liver_mask

    # 5. EXPORT
    final_img = sitk.GetImageFromArray(final_vessels)
    final_img.CopyInformation(ref_ct)
    
    output_path = os.path.join(output_dir, f"ENSEMBLE_{filename}")
    sitk.WriteImage(final_img, output_path)
    print(f"[DONE] Processed: {filename}")

# --- EXECUTION ---
ct_dir = "/path/to/Task08_HepaticVessel/Task08_HepaticVessel/imagesTs"
livs_dir = "/path/to/results"
msd_dir = "/path/to/Documents/Output_Predictions"
out_path = "/path/to/data/ensemble_final"

for case in os.listdir(ct_dir):
    if case.endswith(".nii.gz") and not case.startswith("._"):
        try:
            ensemble_multi_dir(ct_dir, livs_dir, msd_dir, out_path, case)
        except Exception as e:
            print(f"[SKIPPED] {case} due to error: {e}")

