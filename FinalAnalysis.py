import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import pandas as pd
import nibabel as nib

# Graph and Analysis Tools
from skimage.morphology import skeletonize
from skan import Skeleton, csr
import networkx as nx

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def build_global_vessel_graph(binary_mask, spacing_mm):
    """
    Transforms a 3D binary mask into a physically scaled NetworkX graph,
    tracking absolute structural connectivity, loops, and branching paths.
    """
    # 1. Generate 3D Skeleton Line
    skeleton_img = skeletonize(binary_mask.astype(np.uint8))
    
    if np.sum(skeleton_img) == 0:
        return None, 0, 0, 0
        
    # 2. Extract topological graph nodes/edges using physical spacing
    # skan parses the skeleton array into a coordinate skeleton network
    skel_obj = Skeleton(skeleton_img, spacing=spacing_mm)
    
    # 3. Convert to unified NetworkX Graph structure
    G = nx.Graph()
    
    # Parse skan object into graph representations
    # skel_obj.path_coordinates gives us the literal physical mm locations of the paths
    for path_idx in range(skel_obj.n_paths):
        coords = skel_obj.path_coordinates(path_idx)
        # Add nodes with physical mm coordinates as attributes
        start_node = tuple(coords[0])
        end_node = tuple(coords[-1])
        
        # Calculate true physical length of this branch segment in mm
        branch_length = skel_obj.path_lengths()[path_idx]
        
        G.add_node(start_node, pos=start_node)
        G.add_node(end_node, pos=end_node)
        G.add_edge(start_node, end_node, weight=branch_length)
        
    # 4. Calculate Structural Continuity Metrics
    # In a graph, "components" represents broken vessel islands.
    # A graph-trained ensemble (GATSegDiff) should pull this number close to 1.
    num_connected_components = nx.number_connected_components(G)
    
    # Track junction/branching points (nodes with degree > 2)
    degrees = dict(G.degree())
    branch_points = sum(1 for node, deg in degrees.items() if deg > 2)
    end_points = sum(1 for node, deg in degrees.items() if deg == 1)
    
    return G, num_connected_components, branch_points, end_points

def evaluate_and_plot_global_graphs(file_list, output_plot_path="vessel_topology_comparison.png"):
    """
    Automated evaluation harness that scales, processes, builds full 3D graphs, 
    and logs structural continuity vs fragmentation across both architectures.
    """
    graph_metrics_summary = []
    
    print(f"Beginning global 3D graph extraction across {len(file_list)} files...")
    
    for ens_file in file_list:
        filename = Path(ens_file).name.lower()
        
        # 1. Automatically parse physical millimeter spacings
        try:
            nii_img = nib.load(str(ens_file))
            spacing_mm = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
            ensemble_mask = np.asanyarray(nii_img.dataobj) > 0.5
            if len(ensemble_mask.shape) > 3:
                ensemble_mask = ensemble_mask[..., 0]
        except Exception as e:
            print(f"File skipped. Header read issue on {filename}: {e}")
            continue
            
        # 2. Assign Model Architecture Type
        if "gatsegdiff" in filename or "2.5d" in filename:
            model_type = "GATSegDiff Ensemble (2.5D/3D)"
        else:
            model_type = "Pure nnU-Net (3D Baseline)"
            
        # 3. Generate the Complete Vessel Tree Graph
        VesselGraph, fragments, bifurcations, endpoints = build_global_vessel_graph(ensemble_mask, spacing_mm)
        
        if VesselGraph is None:
            print(f"[WARNING] Empty skeleton graph generated for {filename}")
            continue
            
        # Calculate total tracking lengths of the entire parsed vessel graph in mm
        total_vessel_length_mm = sum(d['weight'] for u, v, d in VesselGraph.edges(data=True))
        
        # 4. Save Metrics to Profile Matrix
        graph_metrics_summary.append({
            "File_Name": Path(ens_file).name,
            "Architecture": model_type,
            "Total_Disconnected_Islands": fragments,
            "Total_Bifurcation_Nodes": bifurcations,
            "Vessel_Terminal_Endpoints": endpoints,
            "Total_Tree_Length_mm": round(total_vessel_length_mm, 2),
            # Structural ratio: High numbers point to broken, fragmented voxel islands
            "Fragmentation_Index": round(fragments / max(1, bifurcations), 4)
        })
        
        print(f"[SUCCESS] Processed {model_type} -> Built graph with {bifurcations} forks and {fragments} islands.")

    df_results = pd.DataFrame(graph_metrics_summary)
    
    if df_results.empty:
        print("No graph data extracted. Plotting skipped.")
        return df_results

    # ----------------------------------------------------
    # 5. AUTOMATED STATISTICAL PLOTTING GENERATION
    # ----------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Plot 1: Fragmented Islands Count (lower is better, shows graph continuity)
    sns.violinplot(data=df_results, x="Architecture", y="Total_Disconnected_Islands", palette="Set2", ax=ax1, alpha=0.5)
    sns.boxplot(data=df_results, x="Architecture", y="Total_Disconnected_Islands", palette="Set2", width=0.2, ax=ax1)
    ax1.set_title("Vessel Tree Fragmentation\n(Lower = Better Continuity)", weight='bold', fontsize=12)
    ax1.set_ylabel("Number of Disconnected Graph Components")
    
    # Plot 2: Total Recovered Tree Length in Millimeters (higher is better)
    sns.violinplot(data=df_results, x="Architecture", y="Total_Tree_Length_mm", palette="Set2", ax=ax2, alpha=0.5)
    sns.boxplot(data=df_results, x="Architecture", y="Total_Tree_Length_mm", palette="Set2", width=0.2, ax=ax2)
    ax2.set_title("Total Tracked Vascular Network Length\n(Higher = Better Connectivity)", weight='bold', fontsize=12)
    ax2.set_ylabel("Total Tree Length (mm)")
    
    plt.tight_layout()
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    
    return df_results

# ----------------------------------------------------
# RUN ENGINE
# ----------------------------------------------------
if __name__ == "__main__":
    PREDICTIONS_DIR = "path/to/all_compiled_predictions/" 
    all_nifti_files = [str(p) for p in Path(PREDICTIONS_DIR).glob("*.nii.gz")]
    
    if len(all_nifti_files) > 0:
        df_graph_stats = evaluate_and_plot_global_graphs(all_nifti_files, "global_graph_topology_results.png")
        
        print("\n================== GLOBAL TOPOLOGICAL SUMMARY STATISTICS ==================\n")
        print(df_graph_stats.groupby("Architecture")[[
            "Total_Disconnected_Islands", 
            "Total_Bifurcation_Nodes", 
            "Total_Tree_Length_mm",
            "Fragmentation_Index"
        ]].mean().to_string())
