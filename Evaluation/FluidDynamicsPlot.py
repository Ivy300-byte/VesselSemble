import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader

# 1. Fluid parameters for Womersley number calculation
HEART_RATE_HZ = 1.2      # 72 bpm resting heart rate
KINEMATIC_VISC = 3.3e-6  # m^2/s for blood
OMEGA = 2 * np.pi * HEART_RATE_HZ

# 2. Extract all 6 table blocks sequentially as separate datasets
reader = PdfReader("your_file.pdf")  # <-- Replace with your PDF filename
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

# Split text at each 'case_id' table header (produces 6 distinct blocks)
blocks = re.split(r"case_id\s+Inferred_Inlet_Diam_mm.*?\n", full_text)[1:]

dataset_titles = [
    "Dataset 1: HepaticVessel A (n=140)",
    "Dataset 2: ENSEMBLE Mask (n=20)",
    "Dataset 3: 3Dircadb1 Pred (n=20)",
    "Dataset 4: Patient Cohort A (n=20)",
    "Dataset 5: HepaticVessel B (n=140)",
    "Dataset 6: Patient Cohort B (n=20)",
]

datasets = []
for idx, block in enumerate(blocks):
    # Normalize space-separated IDs on Page 9 ('hepaticvessel 436', 'patient 007 0000')
    block = re.sub(r"\b(hepaticvessel|patient)\s+(\d+)\s+(\d+)\b", r"\1_\2_\3", block)
    block = re.sub(r"\b(hepaticvessel|patient)\s+(\d+)\b", r"\1_\2", block)

    matches = re.findall(
        r"([A-Za-z0-9_.-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        block,
    )
    df_block = pd.DataFrame(
        [
            [cid, float(diam), float(inlet_re), float(branch_re), float(dean), float(pressure)]
            for cid, diam, inlet_re, branch_re, dean, pressure in matches
        ],
        columns=[
            "case_id",
            "Inferred_Inlet_Diam_mm",
            "Inferred_Inlet_Re",
            "Mean_Branch_Re",
            "Max_Dean_Number",
            "Required_Inlet_Pressure_Pa",
        ],
    )
    # Compute Womersley Number (alpha = R * sqrt(omega / nu))
    radius_m = (df_block["Inferred_Inlet_Diam_mm"] * 1e-3) / 2.0
    df_block["Womersley_Alpha"] = radius_m * np.sqrt(OMEGA / KINEMATIC_VISC)
    title = dataset_titles[idx] if idx < len(dataset_titles) else f"Dataset {idx + 1}"
    datasets.append((title, df_block))

# 3. Create a 2x3 grid with one dedicated subplot per dataset
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle(
    f"Separate Reynolds & Womersley (α) Profiles Across All 6 Datasets (f = {HEART_RATE_HZ} Hz)",
    fontsize=15,
    fontweight="bold",
)

for ax, (title, df_sub) in zip(axes.flat, datasets):
    # Plot Inlet Re vs. Womersley Alpha
    ax.scatter(
        df_sub["Womersley_Alpha"],
        df_sub["Inferred_Inlet_Re"],
        color="#1f77b4",
        marker="o",
        s=45,
        alpha=0.85,
        edgecolors="black",
        linewidth=0.4,
        label="Inlet Re",
    )
    # Plot Mean Branch Re vs. Womersley Alpha
    ax.scatter(
        df_sub["Womersley_Alpha"],
        df_sub["Mean_Branch_Re"],
        color="#ff7f0e",
        marker="^",
        s=45,
        alpha=0.85,
        edgecolors="black",
        linewidth=0.4,
        label="Mean Branch Re",
    )

    # Reference line for quasi-steady limit (alpha = 1)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Quasi-steady limit (α = 1)")

    # Summary stats box in top-left of each panel
    stats_text = (
        f"α: {df_sub['Womersley_Alpha'].min():.2f}–{df_sub['Womersley_Alpha'].max():.2f}\n"
        f"Inlet Re: {df_sub['Inferred_Inlet_Re'].min():.0f}–{df_sub['Inferred_Inlet_Re'].max():.0f}\n"
        f"Branch Re: {df_sub['Mean_Branch_Re'].min():.1f}–{df_sub['Mean_Branch_Re'].max():.1f}"
    )
    ax.text(
        0.04, 0.94, stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

# Label outer axes cleanly
for ax in axes[1, :]:
    ax.set_xlabel("Womersley Number (α)", fontsize=10.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Reynolds Number (Re)", fontsize=10.5)

# Single shared legend at the top (safely grab from axes[0, 0])
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True, fontsize=10.5)

plt.savefig("separate_6_datasets_reynolds_womersley.png", dpi=300, bbox_inches="tight")
plt.show()
