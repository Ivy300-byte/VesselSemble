import os
import time
import numpy as np
from pathlib import Path
import pandas as pd
import nibabel as nib
import networkx as nx
from scipy.ndimage import affine_transform
import gudhi

# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

# High-Speed Skeleton Extraction Tools
from skimage.morphology import skeletonize
from skan import Skeleton

def inspect_and_harvest_graphml_keys(graphml_path):
    """
    Automated Metadata Inspector:
    Scans the GraphML file to discover exactly how node coordinates are labeled.
    Prevents silent failures and key guessing.
    """
    print("\n" + "-"*60)
    print("      AUTOMATED METADATA INSPECTOR: HARVESTING KEYS         ")
    print("-"*60)
    
    G_raw = nx.read_graphml(str(graphml_path))
    
    if len(G_raw.nodes) == 0:
        raise ValueError(f"The reference GraphML file '{Path(graphml_path).name}' contains 0 nodes.")
        
    sample_node_id = list(G_raw.nodes)[0]
    sample_data = G_raw.nodes[sample_node_id]
    
    print(f" -> Found total node records: {len(G_raw.nodes)}")
    print(f" -> Sample Node ID Format   : {sample_node_id} (Type: {type(sample_node_id).__name__})")
    print(f" -> Available Key Attributes: {list(sample_data.keys())}")
    
    x_key, y_key, z_key = None, None, None
    
    x_candidates = ['x', 'pos_x', 'pos_X', 'coord_x', 'coord_X', 'X', 'coordinate_x']
    y_candidates = ['y', 'pos_y', 'pos_Y', 'coord_y', 'coord_Y', 'Y', 'coordinate_y']
    z_candidates = ['z', 'pos_z', 'pos_Z', 'coord_z', 'coord_Z', 'Z', 'coordinate_z']
    
    for k in sample_data.keys():
        k_lower = k.lower()
        if k in x_candidates or k_lower == 'x' or 'pos_x' in k_lower or 'coord_x' in k_lower:
            x_key = k
        elif k in y_candidates or k_lower == 'y' or 'pos_y' in k_lower or 'coord_y' in k_lower:
            y_key = k
        elif k in z_candidates or k_lower == 'z' or 'pos_z' in k_lower or 'coord_z' in k_lower:
            z_key = k

    if not (x_key and y_key and z_key):
        print(" -> [NOTICE] Coordinates not found in standard node keys. Checking ID strings...")
        if isinstance(sample_node_id, str) and (',' in sample_node_id or ' ' in sample_node_id):
            print(" -> [SUCCESS] Node IDs appear to be stringified spatial tuples.")
            return "USE_ID_STRINGS", None, None

    if not (x_key and y_key and z_key):
        raise KeyError(f"Could not automatically locate 3D coordinate attributes in your GraphML layout. Keys found: {list(sample_data.keys())}")
        
    print(f" -> [HARVEST SUCCESS] Auto-Mapped Coordinate Keys -> X: '{x_key}', Y: '{y_key}', Z: '{z_key}'")
    print("-"*60 + "\n")
    return x_key, y_key, z_key


def find_3d_bounding_box(binary_mask, padding=2):
    """Finds the absolute coordinate bounds of active voxels in 3D space."""
    if not np.any(binary_mask):
        return None, (0, 0, 0)
    
    x_indices = np.any(binary_mask, axis=(1, 2))
    y_indices = np.any(binary_mask, axis=(0, 2))
    z_indices = np.any(binary_mask, axis=(0, 1))
    
    x_min = int(np.where(x_indices)[0][0])
    x_max = int(np.where(x_indices)[0][-1])
    y_min = int(np.where(y_indices)[0][0])
    y_max = int(np.where(y_indices)[0][-1])
    z_min = int(np.where(z_indices)[0][0])
    z_max = int(np.where(z_indices)[0][-1])
    
    pad_x_min = max(0, x_min - padding)
    pad_y_min = max(0, y_min - padding)
    pad_z_min = max(0, z_min - padding)
    
    pad_x_max = min(binary_mask.shape[0], x_max + padding + 1)
    pad_y_max = min(binary_mask.shape[1], y_max + padding + 1)
    pad_z_max = min(binary_mask.shape[2], z_max + padding + 1)
    
    voxel_offsets = (pad_x_min, pad_y_min, pad_z_min)
    slices = (slice(pad_x_min, pad_x_max), slice(pad_y_min, pad_y_max), slice(pad_z_min, pad_z_max))
    
    return slices, voxel_offsets


def align_and_resample_mask(moving_img, target_shape, target_affine):
    """Rigid resampling engine mapping diverse 2.5D/3D dimensions using NIfTI headers."""
    mapping_affine = np.linalg.inv(moving_img.affine) @ target_affine
    matrix_3d = mapping_affine[:3, :3]
    offset_3d = mapping_affine[:3, 3]
    
    moving_data = np.asanyarray(moving_img.dataobj) > 0.5
    if len(moving_data.shape) > 3:
        moving_data = moving_data[..., 0]
        
    aligned_data = affine_transform(
        moving_data.astype(np.float32), 
        matrix=matrix_3d, 
        offset=offset_3d, 
        output_shape=target_shape, 
        order=0 
    )
    return aligned_data > 0.5


def parse_and_shift_reference_graph(graphml_path, voxel_offsets, spacing_mm, harvested_keys):
    """
    Natively loads the GraphML reference structure, dynamically parsing coordinates
    based on harvested keys, and auto-corrects spatial crop shifts.
    """
    G_raw = nx.read_graphml(str(graphml_path))
    G_shifted = nx.Graph()
    shift_mm = np.array(voxel_offsets, dtype=np.float32) * spacing_mm
    
    x_k, y_k, z_k = harvested_keys
    node_coordinate_cache = {}

    for node, data in G_raw.nodes(data=True):
        if x_k == "USE_ID_STRINGS":
            cleaned_str = str(node).replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            coords = np.fromstring(cleaned_str, sep=',')
            if len(coords) != 3:
                coords = np.fromstring(cleaned_str, sep=' ')
            x, y, z = coords[0], coords[1], coords[2]
        else:
            x = float(data[x_k])
            y = float(data[y_k])
            z = float(data[z_k])
            
        if x_k != "USE_ID_STRINGS" and x < 2000 and float(x).is_integer() and np.any(spacing_mm != 1.0):
             x, y, z = x * spacing_mm[0], y * spacing_mm[1], z * spacing_mm[2]

        adjusted_pos = (x - shift_mm[0], y - shift_mm[1], z - shift_mm[2])
        G_shifted.add_node(adjusted_pos, pos=adjusted_pos)
        node_coordinate_cache[node] = adjusted_pos

    for u, v, data in G_raw.edges(data=True):
        u_adj = node_coordinate_cache[u]
        v_adj = node_coordinate_cache[v]
        w = data.get('weight', data.get('length', np.linalg.norm(np.array(u_adj) - np.array(v_adj))))
        G_shifted.add_edge(u_adj, v_adj, weight=w)

    del G_raw
    node_coordinate_cache.clear()
    return G_shifted


def build_global_vessel_graph(binary_mask, spacing_mm):
    """Transforms a 3D binary mask into a physically scaled NetworkX graph using Lee's thinning."""
    skeleton_img = skeletonize(binary_mask.astype(np.uint8), method='lee')
    if np.sum(skeleton_img) == 0:
        return None, 0, 0, 0

    skel_obj = Skeleton(skeleton_img, spacing=spacing_mm)
    G = nx.Graph()

    path_coordinates_fn = skel_obj.path_coordinates
    path_lengths_array = skel_obj.path_lengths()
    
    for path_idx in range(skel_obj.n_paths):
        coords = path_coordinates_fn(path_idx)
        # Safe coordinate mapping using explicit 1D point endpoints to bypass array issues
        start_node = tuple(coords[0])
        end_node = tuple(coords[-1])
        branch_length = path_lengths_array[path_idx]
        
        G.add_node(start_node, pos=start_node)
        G.add_node(end_node, pos=end_node)
        G.add_edge(start_node, end_node, weight=branch_length)

    num_connected_components = nx.number_connected_components(G)
    degrees = dict(G.degree())
    branch_points = sum(1 for node, deg in degrees.items() if deg > 2)
    end_points = sum(1 for node, deg in degrees.items() if deg == 1)

    return G, num_connected_components, branch_points, end_points


def voxel_grid_downsample(points, voxel_size=2.0):
    """Downsamples 3D point space into a uniform voxel grid to streamline computing."""
    if len(points) == 0:
        return points
    grid_coords = np.floor(points / voxel_size).astype(np.int32)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['g_x'] = grid_coords[:, 0]
    df['g_y'] = grid_coords[:, 1]
    df['g_z'] = grid_coords[:, 2]
    return df.groupby(['g_x', 'g_y', 'g_z'])[['x', 'y', 'z']].mean().values


def compute_vessel_tda_metrics_optimized(G, noise_threshold_mm=1.5, voxel_filter_size=2.0):
    """Computes Alpha Complex persistent homology on graph nodes."""
    if G is None or len(G.nodes) < 2:
        return {"Robust_Loop_Count": 0, "Vessel_Loop_Stability_Score": 0.0, "H1_Intervals": None}

    nodes = list(G.nodes)
    pos_array = np.array([G.nodes[n]['pos'] for n in nodes])
    pos_array = voxel_grid_downsample(pos_array, voxel_size=voxel_filter_size)

    if len(pos_array) < 3:
         return {"Robust_Loop_Count": 0, "Vessel_Loop_Stability_Score": 0.0, "H1_Intervals": None}

    alpha_complex = gudhi.AlphaComplex(points=pos_array)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.compute_persistence()
    intervals_h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if len(intervals_h1) == 0:
        return {"Robust_Loop_Count": 0, "Vessel_Loop_Stability_Score": 0.0, "H1_Intervals": None}

    intervals_h1_mm = np.sqrt(np.clip(intervals_h1, 0, None))
    clean_intervals = [pair for pair in intervals_h1_mm if np.all(np.isfinite(pair))]
    
    if len(clean_intervals) == 0:
         return {"Robust_Loop_Count": 0, "Vessel_Loop_Stability_Score": 0.0, "H1_Intervals": None}
                
    intervals_h1_mm = np.array(clean_intervals)
    loop_lifetimes = intervals_h1_mm[:, 1] - intervals_h1_mm[:, 0]
    robust_loops = loop_lifetimes[loop_lifetimes > noise_threshold_mm]

    return {
        "Robust_Loop_Count": len(robust_loops),
        "Vessel_Loop_Stability_Score": round(float(np.sum(loop_lifetimes)), 2),
        "H1_Intervals": intervals_h1_mm
    }


def compute_graph_distance_accuracy(G_consensus, G_reference, spatial_tolerance_mm=1.5):
    """
    Tolerance-Aware Geometric Jaccard Distance.
    Matches node positions accurately despite tiny floating point variations.
    """
    if G_consensus is None or G_reference is None:
        return 1.0
        
    nodes_c = np.array([G_consensus.nodes[n]['pos'] for n in G_consensus.nodes])
    nodes_r = np.array([G_reference.nodes[n]['pos'] for n in G_reference.nodes])
    
    if len(nodes_c) == 0 or len(nodes_r) == 0:
        return 1.0
        
    diffs = nodes_c[:, None, :] - nodes_r[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    
    matches_c = np.any(dists <= spatial_tolerance_mm, axis=1)
    
    intersection = np.sum(matches_c)
    union = len(nodes_c) + len(nodes_r) - intersection
    
    return round(1.0 - (intersection / union), 4) if union > 0 else 0.0


def plot_comparative_persistence(c_intervals, r_intervals, output_path, noise_cutoff=1.5):
    """Plots the consensus and reference loops side-by-side on one diagram."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5.5), dpi=100)
    
    max_val = 5.0
    if c_intervals is not None and len(c_intervals) > 0:
        c_lifetimes = c_intervals[:, 1] - c_intervals[:, 0]
        c_filtered = c_intervals[c_lifetimes > noise_cutoff]
        if len(c_filtered) > 0:
            plt.scatter(c_filtered[:, 0], c_filtered[:, 1], color="darkblue", alpha=0.7, s=50, label="Consensus Loops", zorder=3)
            max_val = max(max_val, np.max(c_filtered))
            
    if r_intervals is not None and len(r_intervals) > 0:
        r_lifetimes = r_intervals[:, 1] - r_intervals[:, 0]
        r_filtered = r_intervals[r_lifetimes > noise_cutoff]
        if len(r_filtered) > 0:
            plt.scatter(r_filtered[:, 0], r_filtered[:, 1], color="forestgreen", alpha=0.7, s=50, marker="X", label="Ground-Truth Loops", zorder=3)
            max_val = max(max_val, np.max(r_filtered))
        
    plt.plot([0, max_val + 2], [0, max_val + 2], 'k--', alpha=0.5, label="Identity Diagonal")
    x_space = np.linspace(0, max_val + 2, 100)
    plt.fill_between(x_space, x_space, x_space + noise_cutoff, color='gray', alpha=0.15, label="Filtered Noise Floor")

    plt.title("2.5D/3D Consensus vs Reference Topology Mode", weight='bold', fontsize=11)
    plt.xlabel("Loop Birth Scale (mm)")
    plt.ylabel("Loop Dissolution/Death Scale (mm)")
    plt.legend(loc="lower right", fontsize=9)
    plt.xlim(0, max_val + 1)
    plt.ylim(0, max_val + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def run_full_testing_pipeline(file_list, reference_graphml, plot_path, vote_threshold=0.5, voxel_filter=2.0):
    """
    METADATA-DRIVEN VALIDATION ENGINE (ADAPTIVE AUTO-THRESHOLDING)
    """
    start_time = time.perf_counter()
    harvested_keys = inspect_and_harvest_graphml_keys(reference_graphml)
    
    print(f"Executing adaptive testing pipeline across {len(file_list)} 2.5D/3D files...")
    
    ref_img = nib.load(str(file_list[0]))
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape[:3]
    spacing_mm = np.array(ref_img.header.get_zooms()[:3], dtype=np.float32)
    
    accumulated_counts = np.zeros(ref_shape, dtype=np.float32)
    valid_file_count = 0

    for idx, ens_file in enumerate(file_list):
        filename = Path(ens_file).name
        try:
            img = nib.load(str(ens_file))
            if img.shape[:3] == ref_shape and np.allclose(img.affine, ref_affine, atol=1e-3):
                mask_data = np.asanyarray(img.dataobj).astype(np.float32)
                if mask_data.max() > 1.0: 
                    mask_data = mask_data / 255.0
                if len(mask_data.shape) > 3: mask_data = mask_data[..., 0]
                accumulated_counts += (mask_data > 0.5)
            else:
                resampled_mask = align_and_resample_mask(img, ref_shape, ref_affine)
                accumulated_counts += resampled_mask.astype(np.float32)
            valid_file_count += 1
        except Exception as e:
            print(f" -> [SKIPPED] Error re-aligning {filename}: {e}")
            continue

    max_agreement = float(np.max(accumulated_counts))
    print(f" -> [ENSEMBLE METADATA] Peak voxel overlay count: {max_agreement} models overlapping.")
    
    scaled_threshold_count = vote_threshold * valid_file_count
    if max_agreement < scaled_threshold_count:
        adaptive_count = max(1.0, max_agreement * 0.15)
        print(f" -> [AUTO-OPTIMIZATION] Vote threshold of {scaled_threshold_count} files would empty the mask.")
        print(f" -> [AUTO-OPTIMIZATION] Adaptively lowering vote requirement to {adaptive_count:.1f} file overlaps.")
        consensus_mask = accumulated_counts >= adaptive_count
    else:
        consensus_mask = accumulated_counts >= scaled_threshold_count
        
    del accumulated_counts 

    # 2. Extract 3D Bounding-Box Crops
    bbox_slices, voxel_offsets = find_3d_bounding_box(consensus_mask)
    if bbox_slices is not None:
        consensus_mask = consensus_mask[bbox_slices]
        print(f" -> [BBOX COMPLETE] Isolated active spatial matrix shape to: {consensus_mask.shape}")
    else:
        print(" -> [CRITICAL WARNING] Consensus mask is completely blank.")

    # 3. Optimized Skeleton-Based Graph Extraction
    print("Extracting centerline paths for consensus network graph...")
    G_consensus, c_frag, c_bif, c_end = build_global_vessel_graph(consensus_mask, spacing_mm)
    c_length = sum(d['weight'] for u, v, d in G_consensus.edges(data=True)) if G_consensus else 0.0

    # 4. Process & Realignment of Reference GraphML Coordinate Matrix
    G_reference = None
    ref_length = 0.0
    ref_tda = {"Robust_Loop_Count": 0, "Vessel_Loop_Stability_Score": 0.0, "H1_Intervals": None}
    
    if reference_graphml and os.path.exists(reference_graphml):
        G_reference = parse_and_shift_reference_graph(reference_graphml, voxel_offsets, spacing_mm, harvested_keys)
        ref_length = sum(d['weight'] for u, v, d in G_reference.edges(data=True)) if G_reference else 0.0
        ref_tda = compute_vessel_tda_metrics_optimized(G_reference, voxel_filter_size=voxel_filter)

    # 5. Fast Alpha Complex Persistent Homology Analysis on the Consensus Matrix
    print("Calculating Alpha Complex persistent homology on consensus data...")
    consensus_tda = compute_vessel_tda_metrics_optimized(G_consensus, voxel_filter_size=voxel_filter)
    
    # 6. Quantify Structural/Topological Errors using Tolerance Matching
    graph_error = compute_graph_distance_accuracy(G_consensus, G_reference, spatial_tolerance_mm=1.5)

    # 7. Print Final Metrics Summary Matrix Dashboard
    print("\n" + "="*85)
    print("              FINAL 2.5D/3D ENSEMBLE VALIDATION MATRIX DETAILS               ")
    print("="*85)
    print(f" -> Geometric Graph Jaccard Error   : {graph_error} (Lower = Better)")
    print("-"*85)
    print(f" METRIC DESCRIPTOR                  | CONSENSUS       | REFERENCE GROUND-TRUTH")
    print(f" - Disconnected Islands (Fragments) : {c_frag:<15} | {nx.number_connected_components(G_reference) if G_reference else 0}")
    print(f" - True Vessel Loops (Robust Count) : {consensus_tda['Robust_Loop_Count']:<15} | {ref_tda['Robust_Loop_Count']}")
    print(f" - Cumulative Topological Stability : {consensus_tda['Vessel_Loop_Stability_Score']:<15} | {ref_tda['Vessel_Loop_Stability_Score']}")
    print(f" - Total Network Tree Length (mm)   : {round(c_length, 2):<15} | {round(ref_length, 2)}")
    print("="*85)
    print(f"Total CPU Processing Evaluation complete in: {time.perf_counter() - start_time:.2f} seconds.\n")

    # Generate visual summary diagnostic charts
    if consensus_tda["H1_Intervals"] is not None or ref_tda["H1_Intervals"] is not None:
         plot_comparative_persistence(consensus_tda["H1_Intervals"], ref_tda["H1_Intervals"], plot_path, noise_cutoff=1.5)
    
    summary_data = {
        "Graph_Jaccard_Error": graph_error,
        "Consensus_Islands": c_frag,
        "Consensus_Loops": consensus_tda['Robust_Loop_Count'],
        "Consensus_Stability": consensus_tda['Vessel_Loop_Stability_Score'],
        "Consensus_Length_mm": round(c_length, 2)
    }
    return consensus_mask, ref_affine, summary_data


# --------------------------------------------------------------------
# SYSTEM ENGINE INITIALIZATION
# --------------------------------------------------------------------
if __name__ == "__main__":
    # YOUR SYSTEM EXPLICIT ENDPOINTS (Using raw string r"..." formatting)
    INPUT_PREDICTIONS_DIR = r"C:\Users\ifeeu\OneDrive\Documents\data_1\masks_final_1_1_2_1"
    GROUND_TRUTH_GRAPHML = r"C:\Users\ifeeu\Downloads\vessel_plus_recursive_network.graphml"
    
    OUTPUT_MASK_NAME = "assembled_2.5d_3d_consensus.nii.gz"
    OUTPUT_TDA_CHART = "validation_tda_comparison.png"
    
    OUTPUT_CSV_DIR = r"C:\Users\ifeeu\Downloads\topological_metrics_logs"
    OUTPUT_CSV_FILENAME = "validation_summary_metrics.csv"
    
    VOTING_RULE_THRESHOLD = 0.5 
    VOXEL_GRID_FILTER_SIZE = 2.0  

    input_path_obj = Path(INPUT_PREDICTIONS_DIR)
    all_nifti_files = sorted([str(p) for p in input_path_obj.glob("*.nii.gz")])

    if len(all_nifti_files) == 0:
        print(f"[INITIALIZATION FAILURE] No target NIfTI volumes found inside: '{input_path_obj.resolve()}'")
    else:
        # Run entire validation workflow
        final_mask, out_affine, metrics_dict = run_full_testing_pipeline(
            all_nifti_files, 
            GROUND_TRUTH_GRAPHML, 
            OUTPUT_TDA_CHART, 
            VOTING_RULE_THRESHOLD, 
            VOXEL_GRID_FILTER_SIZE
        )
        
        # 1. Save out the unified consensus segmentation NIfTI matrix
        print(f"Saving compiled NIfTI voxel mask data...")
        ref_img_dims = nib.load(all_nifti_files[0])
        output_nii = nib.Nifti1Image(final_mask.astype(np.uint8), out_affine)
        output_nii.header.set_zooms(np.array(ref_img_dims.header.get_zooms()[:3], dtype=np.float32))
        nib.save(output_nii, OUTPUT_MASK_NAME)
        print(f" -> Consensus mask exported successfully: {OUTPUT_MASK_NAME}")
        
        # 2. Automated path validation & resolution tracking for CSV files
        csv_dir_path = Path(OUTPUT_CSV_DIR)
        csv_dir_path.mkdir(parents=True, exist_ok=True) 
        full_csv_save_path = csv_dir_path / OUTPUT_CSV_FILENAME
        
        pd.DataFrame([metrics_dict]).to_csv(full_csv_save_path, index=False)
        print(f" -> Quantitative metric log updated successfully at: {full_csv_save_path}")
