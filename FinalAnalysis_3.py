import os, psutil, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx

def get_optimal_energy_workers():
    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    return max(1, int(physical_cores * 0.75))

def load_reference_graph_with_metadata(graphml_path, ensemble_input_dir):
    print(f"\n[1/5] Loading global reference 3D graph: {Path(graphml_path).name}")
    G = nx.read_graphml(graphml_path)
    img = nib.load(str(next(Path(ensemble_input_dir).glob("*.nii*"))))
    zooms = img.header.get_zooms()[:3]
    return G, {
        'spacing_mm': [float(z) for z in zooms],
        'matrix_size': list(img.shape),
        'voxel_tolerance': float(np.mean(zooms) * 3.0)
    }

def process_single_mask(mask_file, threshold=0.5):
    try:
        img = nib.load(str(mask_file))
        data = img.get_fdata(dtype=np.float32)
        if data.max() == data.min(): return np.empty((0, 3))
        
        data = (data - data.min()) / (data.max() - data.min() + 1e-7)
        active = np.argwhere(data >= threshold)
        if len(active) == 0: return np.empty((0, 3))
        
        homogenous = np.hstack([active[:, :3], np.ones((len(active), 1))])
        return (img.affine @ homogenous.T).T[:, :3]
    except Exception as e:
        print(f" -> Skip file {mask_file.name}: {e}")
        return np.empty((0, 3))

def fuse_ensemble_masks_with_metadata(ensemble_dir, metadata, fusion_threshold=0.5, workers=4):
    mask_files = list(Path(ensemble_dir).glob("*.nii*"))
    print(f"\n[2/5] Synthesizing {len(mask_files)} components in world space...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda f: process_single_mask(f, fusion_threshold), mask_files))
        
    valid_coords = [res for res in results if len(res) > 0]
    if not valid_coords: return np.empty((0, 3)), np.array(metadata['matrix_size'])
        
    fused_coords = np.unique(np.round(np.vstack(valid_coords), decimals=1), axis=0)
    print(f" -> Active Prediction Coordinates Extracted: {len(fused_coords)}")
    return fused_coords, fused_coords.max(axis=0)

def parse_positions_from_graph(G):
    """
    Open-ended schema-free parser. Extracts all numbers found in node IDs or properties
    and converts them to 3D matrix space, dynamically zero-padding any format.
    """
    coords = []
    num_pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    
    for node, data in G.nodes(data=True):
        raw_values = []
        
        # Collect every string value associated with this node's properties
        for value in data.values():
            if value is not None:
                raw_values.extend(num_pattern.findall(str(value)))
                
        # If properties contain no numeric entities, scan the primary Node ID string label
        if not raw_values:
            raw_values = num_pattern.findall(str(node))
            
        if raw_values:
            # Convert text entries to a float sequence
            float_vals = [float(num) for num in raw_values]
            
            # Map into 3D structure: Slice to length 3, or pad out using zero indices if shorter
            if len(float_vals) >= 3:
                coords.append(float_vals[:3])
            else:
                padded_vals = float_vals + [0.0] * (3 - len(float_vals))
                coords.append(padded_vals)
                
    return np.array(coords).reshape(-1, 3) if coords else np.empty((0, 3))

def evaluate_spatial_metrics(gt_landmarks, predicted_points, metadata, workers=4):
    total_gt, total_pred = len(gt_landmarks), len(predicted_points)
    if total_gt == 0:
        return pd.DataFrame([{
            "Vessel Class": "hybrid_ensemble", "GT Landmarks": 0, "Total Predicted Points": total_pred,
            "Strict Hits (<Tol)": 0, "Proximity Halo Points": 0, "Distant Outliers": total_pred,
            "Mean Distance (mm)": 0.00, "Landmark Recall %": 0, "Precision %": 0
        }])
        
    distances, _ = KDTree(gt_landmarks).query(predicted_points, workers=workers)
    tol = metadata['voxel_tolerance']
    gt_distances, _ = KDTree(predicted_points).query(gt_landmarks, workers=workers)
    
    return pd.DataFrame([{
        "Vessel Class": "hybrid_ensemble", "GT Landmarks": total_gt, "Total Predicted Points": total_pred,
        "Strict Hits (<Tol)": int(np.sum(distances <= tol)), 
        "Proximity Halo Points": int(np.sum((distances > tol) & (distances <= tol * 3))), 
        "Distant Outliers": int(np.sum(distances > tol * 3)),
        "Mean Distance (mm)": round(np.mean(distances), 2), 
        "Landmark Recall %": int((np.sum(gt_distances <= tol * 3) / total_gt) * 100), 
        "Precision %": int((np.sum(distances <= tol) / total_pred) * 100)
    }])

def plot_vascular_alignment_overlays(gt_landmarks, predicted_points, calculated_bounds, output_dir):
    print(f"\n[5/5] Generating 3D Structural Alignment Matplotlib Figure...")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(predicted_points) > 0:
        p_disp = predicted_points[np.random.choice(len(predicted_points), min(8000, len(predicted_points)), replace=False)]
        ax.scatter(p_disp[:, 0], p_disp[:, 1], p_disp[:, 2], color='#f6a2b1', s=2, alpha=0.5, label='Predictions')
    if len(gt_landmarks) > 0:
        ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], gt_landmarks[:, 2], color='#1f519b', s=45, edgecolors='black', label='GT Landmarks')
        
    all_bounds = np.vstack([gt_landmarks, predicted_points]) if len(gt_landmarks) > 0 else predicted_points
    max_xyz = float(all_bounds.max()) if len(all_bounds) > 0 else 300
    
    ax.set_xlim(0, max_xyz * 1.05); ax.set_ylim(0, max_xyz * 1.05); ax.set_zlim(0, max_xyz * 1.05)
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("Global Unified Vascular Network 3D Coordinate Mapping Chart", fontweight='bold', pad=15)
    plt.savefig(Path(output_dir) / "global_alignment_overlay.png", dpi=200, bbox_inches='tight')
    plt.close()

# --- Configurations ---
GLOBAL_GRAPH_FILE = Path(r"C:\Users\ifeeu\Downloads\vessel_plus_recursive_network.graphml")
ENSEMBLE_INPUT_DIR = Path(r"C:\Users\ifeeu\OneDrive\Documents\data_2\masks_final_2_1")
OUTPUT_DIR = Path(r"C:\Users\ifeeu\OneDrive\Documents\data_2\masks_final_2_1\graph_results")

# --- Execution ---
try:
    G_ref, metadata = load_reference_graph_with_metadata(GLOBAL_GRAPH_FILE, ENSEMBLE_INPUT_DIR)
    gt_junction_coords = parse_positions_from_graph(G_ref)
    print(f" -> Parsed Graph Nodes: Derived {gt_junction_coords.shape} tracking elements.")
    
    budgeted_workers = get_optimal_energy_workers()
    fused_predicted_coords, calculated_bounds = fuse_ensemble_masks_with_metadata(ENSEMBLE_INPUT_DIR, metadata, 0.50, budgeted_workers)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(OUTPUT_DIR) / "summary.txt", "w") as f: f.write("LIVER-DIR-QA METRICS\n")
    
    analysis_df = evaluate_spatial_metrics(gt_junction_coords, fused_predicted_coords, metadata, budgeted_workers)
    analysis_df.to_csv(OUTPUT_DIR / "global_vascular_analysis_metrics.csv", index=False)
    
    plot_vascular_alignment_overlays(gt_junction_coords, fused_predicted_coords, calculated_bounds, OUTPUT_DIR)
    print(f"\n[Success] Execution successfully completed:\n{analysis_df.to_string(index=False)}")
except Exception as e:
    print(f"\n[Execution Failure] Broken loop sequence: {e}")
