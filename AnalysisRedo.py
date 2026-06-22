import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from skan import Skeleton, summarize
from gtda.homology import VietorisRipsPersistence
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def analyze_single_case(file_path, model_name):
    filename = os.path.basename(file_path)
    try:
        # 1. Optimize Memory and Data Type Loading
        nii_img = nib.load(file_path)
        mask = np.asanyarray(nii_img.dataobj).astype(np.uint8)
        voxel_spacing = nii_img.header.get_zooms()

        # DYNAMIC FIX: Remove small floating background noise artifacts (< 15 voxels)
        # This keeps massive, messy predictions from freezing the skeleton engine.
        mask_labeled = label(mask)
        mask = remove_small_objects(mask_labeled, min_size=15).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)

        # 2. Graphical Engine (Skeletonization)
        skeleton = skeletonize(mask)
        if np.sum(skeleton) == 0:
            geo_metrics = {
                "total_vessel_length_mm": 0.0,
                "bifurcation_count": 0,
                "mean_tortuosity": 1.0,
                "average_branch_length_mm": 0.0
            }
        else:
            skan_skeleton = Skeleton(skeleton, spacing=voxel_spacing)
            branches = summarize(skan_skeleton)
            
            if 'branch-distance' in branches.columns:
                dist_col = 'branch-distance'
            elif 'main-branch-distance' in branches.columns:
                dist_col = 'main-branch-distance'
            else:
                dist_col = 'path-length'

            branches['tortuosity'] = branches[dist_col] / np.where(
    branches['euclidean-distance'] == 0, 1, branches['euclidean-distance']
)
            geo_metrics = {
                "total_vessel_length_mm": round(branches[dist_col].sum(), 2),
                "bifurcation_count": int(np.sum(skan_skeleton.degrees > 2)),
                "mean_tortuosity": round(branches['tortuosity'].mean(), 3),
                "average_branch_length_mm": round(branches[dist_col].mean(), 2)
            }

        # 3. Dynamic TDA Engine (Topological Data Analysis)

        vessel_coords = np.argwhere(skeleton > 0) 
        num_points = len(vessel_coords)

        if num_points < 10:
           topo_metrics = {"fragmentation_errors_h0": 0, "false_loops_h1": 0}

        else:
            # DYNAMIC FIX: Aggressive point ceiling & striding for massive files.
            # If a file is huge, we slice it with a step factor first to save RAM before sampling.
            if num_points > 10000:
                stride_step = max(1, num_points // 5000)
                vessel_coords = vessel_coords[::stride_step]
                num_points = len(vessel_coords)

            # Cap absolute maximum TDA points to 800 for instant algebraic processing
            target_points = min(800, max(300, int(num_points * 0.05)))
            if num_points > target_points:
                indices = np.random.choice(num_points, target_points, replace=False)
                vessel_coords = vessel_coords[indices]
                
            vr = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=1)
            diagrams = vr.fit_transform([vessel_coords]) 
            diagram = diagrams[0]
            
            lifespans = diagram[:, 1] - diagram[:, 0]
            significant_features = diagram[lifespans > 2.0]
            
            h0_count = np.sum(significant_features[:, 2] == 0)
            h1_count = np.sum(significant_features[:, 2] == 1)
            
            topo_metrics = {
                "fragmentation_errors_h0": max(0, int(h0_count - 1)),
                "false_loops_h1": int(h1_count)
            }

        return {
            "case_id": filename.replace(".nii.gz", ""),
            "model_architecture": model_name,
            **geo_metrics,
            **topo_metrics,
            "status": "Success"
        }
        
    except Exception as e:
        error_msg = str(e)
        if "branches" in locals() and isinstance(branches, pd.DataFrame):
            error_msg += f" | Skan Columns: {list(branches.columns)}"
        return {
            "case_id": filename.replace(".nii.gz", ""),
            "model_architecture": model_name,
            "status": f"Failed: {error_msg}"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Democratized Unsupervised Vascular Auditor")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to your model predictions folder")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model architecture")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to save the compiled metrics")
    args = parser.parse_args()

    file_paths = [os.path.join(args.pred_dir, f) for f in os.listdir(args.pred_dir) if f.endswith('.nii.gz')]
    
    if not file_paths:
        print(f"Error: No NIfTI predictions found in {args.pred_dir}")
        exit()

    print(f"Running adaptive-scale pipeline for framework: {args.model_name}")
    results = []
    
    # Restricting multi-processing workers protects your system from running out of memory
    max_workers = min(4, max(1, os.cpu_count() - 1))
    print(f"Spawning pool with {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_case, fp, args.model_name): fp for fp in file_paths}
        
        for f in tqdm(as_completed(futures), total=len(file_paths), desc=f"Auditing {args.model_name}"):
            results.append(f.result())

    df = pd.DataFrame(results)
    
    if 'status' in df.columns:
        status_col = df.pop('status')
        df.insert(2, 'status', status_col)
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model_name.lower()}_unsupervised_audit.csv")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Audit completed. Report saved to: {output_path}")
    except PermissionError:
        alternative_path = os.path.join(args.output_dir, f"{args.model_name.lower()}_unsupervised_audit_fallback.csv")
        df.to_csv(alternative_path, index=False)
        print(f"Permission Blocked! File saved to fallback name instead: {alternative_path}")
