import os
import argparse
import SimpleITK as sitk
import numpy as np
import gc

def post_process_ensemble(input_dir, output_dir):
    """Automates thresholding and cleans up noise while ensuring vessels are preserved."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz") and not f.startswith("._")]
    
    print(f"Processing {len(files)} ensemble maps...\n")
    
    for filename in files:
        # Initialize variables as None so 'del' doesn't crash
        prob_img = label_mask = binary_img = final_mask = None
        
        try:
            prob_path = os.path.join(input_dir, filename)
            # Read and ensure float type for thresholding
            prob_img = sitk.Cast(sitk.ReadImage(prob_path), sitk.sitkFloat32)
            
            # Check if image has data
            stats = sitk.StatisticsImageFilter()
            stats.Execute(prob_img)
            max_val = stats.GetMaximum()
            
            if max_val < 1e-5:
                print(f"[SKIP] {filename} is empty (Max: {max_val})")
                continue

            # 2. Multi-Otsu Thresholding
            try:
                multi_otsu = sitk.MultiOtsuThresholdImageFilter()
                multi_otsu.SetNumberOfThresholds(2)
                label_mask = multi_otsu.Execute(prob_img)
                binary_img = sitk.NotEqual(label_mask, 0)
                thresh_used = multi_otsu.GetThresholds()[0]
            except Exception:
                print(f"[FALLBACK] {filename} - Using fixed threshold 0.5")
                binary_img = sitk.BinaryThreshold(prob_img, lowerThreshold=0.5, upperThreshold=1.0)
                thresh_used = 0.5

            # 3. Noise Cleaning (Keep top 100 clusters)
            cc = sitk.ConnectedComponent(binary_img)
            relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
            final_mask = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=100)

            # 4. Export
            stats.Execute(sitk.Cast(final_mask, sitk.sitkFloat32))
            voxel_count = stats.GetSum()
            
            if voxel_count > 0:
                final_mask.CopyInformation(prob_img)
                out_name = filename if filename.startswith("FINAL_") else f"FINAL_{filename}"
                sitk.WriteImage(sitk.Cast(final_mask, sitk.sitkUInt8), os.path.join(output_dir, out_name))
                print(f"[SUCCESS] {filename} | Thresh: {thresh_used:.4f} | Voxels: {int(voxel_count)}")
            else:
                print(f"[EMPTY] {filename} - No voxels survived.")

        except Exception as e:
            print(f"[ERROR] Failed {filename}: {e}")
        
        finally:
            # Safe memory cleanup
            del prob_img, label_mask, binary_img, final_mask
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    post_process_ensemble(args.input_dir, args.output_dir)

