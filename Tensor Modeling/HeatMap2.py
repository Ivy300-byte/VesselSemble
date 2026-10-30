import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. PARSE CRITICAL SUBSPACE SUITE DATA
# ==============================================================================
models = ["nnUNet", "SimpleUNet", "GATSegDiff"]

# Metrics extracted directly from the autopsy reports
collapsed_counts = [166, 25, 63]
stable_clean_counts = [384, 35, 199]
anomaly_counts = [0, 0, 8]
total_patches = [550, 60, 270]

# Normalize to percentages to see structural footprints relative to pool sizes
collapsed_pct = [c / t * 100 for c, t in zip(collapsed_counts, total_patches)]
stable_pct = [s / t * 100 for s, t in zip(stable_clean_counts, total_patches)]
anomaly_pct = [a / t * 100 for a, t in zip(anomaly_counts, total_patches)]

# ==============================================================================
# 2. CONSTRUCT GENERATIVE SUBSPACE HEATMAPS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle(
    "Linear Algebra Autopsy: Jacobian Subspace Fingerprints",
    fontsize=16,
    fontweight="bold",
    y=1.05,
)

# Custom categorical colormap: Collapsed (Red), Stable Clean (Green), Anomaly (Purple)
categories = [
    "Collapsed Subspaces\n(Det ≈ 0)",
    "Stable Spaces\n(Lossless Projections)",
    "Numerical Anomalies\n(Unclean λ)",
]
colors = ["#e74c3c", "#2ecc71", "#9b59b6"]

for i, ax in enumerate(axes):
  # Consolidate categorical percentages into a 1D column vector layout
  heatmap_data = np.array([[collapsed_pct[i]], [stable_pct[i]], [anomaly_pct[i]]])

  # Draw the matrix configuration representation
  im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto", vmin=0, vmax=100)

  # Configure structural grid aesthetics
  ax.set_title(f"{models[i]}\n(Pool: {total_patches[i]} Patches)", fontsize=12, fontweight="bold")
  ax.set_xticks([])
  ax.set_yticks(range(3))
  ax.set_yticklabels(categories, fontsize=10, fontweight="semibold")

  # Inject data annotations showing the raw counts alongside percentages
  raw_counts = [collapsed_counts[i], stable_clean_counts[i], anomaly_counts[i]]
  pct_vals = [collapsed_pct[i], stable_pct[i], anomaly_pct[i]]

  for row in range(3):
    text_color = "white" if pct_vals[row] < 40 else "black"
    ax.text(
        0,
        row,
        f"{raw_counts[row]} Patches\n({pct_vals[row]:.1f}%)",
        ha="center",
        va="center",
        color=text_color,
        fontsize=11,
        fontweight="bold",
    )

# Establish uniform global colorbar formatting
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(
    "Subspace Occupancy Footprint (%)", fontsize=12, fontweight="bold", labelpad=10
)

plt.subplots_adjust(wspace=0.15)
plt.show()

