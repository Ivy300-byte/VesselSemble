import os
import SimpleITK as sitk
import numpy as np

# --- CONFIG ---
ct_nifti_root = "/home/fader/CT_Nifti"
livs_dir = "/home/fader/predictions"
msd_dir = "/home/fader/Output_Predictions"
out_path = "/home/fader/data_2/ensemble_final_2"

# Manually mapped MSD filenames
msd_filename_list = [
    "hepaticvessel_003.nii.gz", "hepaticvessel_009.nii.gz", "hepaticvessel_012.nii.gz",
    "hepaticvessel_014.nii.gz", "hepaticvessel_015.nii.gz", "hepaticvessel_017.nii.gz",
    "hepaticvessel_021.nii.gz", "hepaticvessel_024.nii.gz", "hepaticvessel_034.nii.gz",
    "hepaticvessel_035.nii.gz", "hepaticvessel_036.nii.gz", "hepaticvessel_037.nii.gz",
    "hepaticvessel_038.nii.gz", "hepaticvessel_041.nii.gz", "hepaticvessel_043.nii.gz",
    "hepaticvessel_045.nii.gz", "hepaticvessel_047.nii.gz", "hepaticvessel_048.nii.gz",
    "hepaticvessel_054.nii.gz", "hepaticvessel_055.nii.gz",
]

def load_and_align(path, reference, is_msd=False):
    img = sitk.ReadImage(path)
    
    # Force the prediction to occupy the same physical coordinates as the CT.
    # This prevents 'Source image size mismatch' errors and ensures overlap.
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    img.SetDirection(reference.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0) # Pad missing slices with zero
    
    # Resampler handles the slice count difference (e.g., 129 vs 124)
    aligned_img = resampler.Execute(img)
    arr = sitk.GetArrayFromImage(aligned_img)
    
    if is_msd:
        return (arr == 1).astype(np.float32)
    return (arr > 0).astype(np.float32)

os.makedirs(out_path, exist_ok=True)
livs_files = [f for f in os.listdir(livs_dir) if f.lower().endswith(".nii.gz")]

print(f"Startup: Found {len(livs_files)} LiVS files and 20 MSD entries.")

for i in range(1, 21):
    patient_id = f"3Dircadb1.{i}"
    idx = i - 1
    
    if idx >= len(msd_filename_list):
        continue
        
    msd_file_name = msd_filename_list[idx]
    msd_path = os.path.join(msd_dir, msd_file_name)
    ct_volume_path = os.path.join(ct_nifti_root, f"{patient_id}.nii.gz")
    file_25d = next((f for f in livs_files if patient_id in f), None)

    if not os.path.exists(ct_volume_path) or not os.path.exists(msd_path) or not file_25d:
        print(f"[SKIP] {patient_id}: Missing file(s).")
        continue

    try:
        print(f"[PROCESS] {patient_id} <-> {msd_file_name}")
        ref_ct = sitk.ReadImage(ct_volume_path)
        
        # Load and Align (Handling slice mismatches)
        arr_25d = load_and_align(os.path.join(livs_dir, file_25d), ref_ct, is_msd=False)
        arr_3d = load_and_align(msd_path, ref_ct, is_msd=True)

        # Weighted Fusion
        fused = (arr_25d * 0.75) + (arr_3d * 0.25)
        
        # Threshold (0.2 ensures we don't zero out data where only one model is present)
        final_mask = (fused > 0.2).astype(np.uint8) 

        print(f"  -> Resulting voxels: {np.count_nonzero(final_mask)}")

        # Save the final result
        final_img = sitk.GetImageFromArray(final_mask)
        final_img.CopyInformation(ref_ct)
        sitk.WriteImage(final_img, os.path.join(out_path, f"ENSEMBLE_{patient_id}.nii.gz"))
        
    except Exception as e:
        print(f"[ERROR] {patient_id}: {e}")

print("\nProcessing complete.")

