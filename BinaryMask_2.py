import os
import argparse
import SimpleITK as sitk
import gc

def post_process_ensemble(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz") and not f.startswith(".")]
    print(f"Processing {len(files)} maps with recovery logic...\n")

    for filename in files:
        prob_img = label_mask = binary_img = final_mask = None
        try:
            prob_path = os.path.join(input_dir, filename)
            prob_img = sitk.Cast(sitk.ReadImage(prob_path), sitk.sitkFloat32)

            # 1. Intensity Analysis
            stats = sitk.StatisticsImageFilter()
            stats.Execute(prob_img)
            max_val = stats.GetMaximum()
            
            if max_val < 1e-5:
                print(f"[SKIP] {filename} is completely empty.")
                continue

            # 2. Adaptive Thresholding
            try:
                # Try Multi-Otsu for high-contrast images
                multi_otsu = sitk.MultiOtsuThresholdImageFilter()
                multi_otsu.SetNumberOfThresholds(2) 
                label_mask = multi_otsu.Execute(prob_img)
                binary_img = sitk.Greater(label_mask, 0)
                thresh_info = f"Multi-Otsu ({multi_otsu.GetThresholds()[0]:.4f})"
            except:
                # RECOVERY: Use 20% of the max intensity if Multi-Otsu fails
                recov_thresh = max_val * 0.2
                binary_img = sitk.BinaryThreshold(prob_img, recov_thresh, 1.0, 1, 0)
                thresh_info = f"Recovery-Adaptive ({recov_thresh:.4f})"

            # 3. Component-Based Noise Cleaning
            cc = sitk.ConnectedComponent(binary_img)
            # Sort by size: Label 1 is largest, 2 is second largest, etc.
            relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
            
            # SAFE FILTER: Keep the 500 largest components instead of 100
            # This prevents deleting fragmented vessel branches
            final_mask = sitk.BinaryThreshold(relabeled, 1, 500, 1, 0)

            # 4. Save and Report
            stats.Execute(sitk.Cast(final_mask, sitk.sitkFloat32))
            voxel_count = int(stats.GetSum())

            if voxel_count > 0:
                final_mask.CopyInformation(prob_img)
                out_name = filename if filename.startswith("FINAL_") else f"FINAL_{filename}"
                sitk.WriteImage(sitk.Cast(final_mask, sitk.sitkUInt8), os.path.join(output_dir, out_name))
                print(f"[SUCCESS] {filename} | {thresh_info} | Voxels: {voxel_count}")
            else:
                # FINAL FALLBACK: If cleaning killed everything, save the raw binary mask
                final_mask = binary_img
                final_mask.CopyInformation(prob_img)
                sitk.WriteImage(sitk.Cast(final_mask, sitk.sitkUInt8), os.path.join(output_dir, f"RAW_{filename}"))
                print(f"[RECOVERED] {filename} - Cleaning was too aggressive, saved RAW mask.")

        except Exception as e:
            print(f"[ERROR] Failed {filename}: {e}")
        finally:
            del prob_img, label_mask, binary_img, final_mask
            gc.collect()

if __name__ == "__main__":
    # Using your specific paths as defaults
    input_path = "/home/fader/data_1/ensemble_final_1"
    output_path = "/home/fader/data/masks_final_1_1_2_1"
    post_process_ensemble(input_path, output_path)

