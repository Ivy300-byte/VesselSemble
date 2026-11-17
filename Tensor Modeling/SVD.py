import matplotlib.pyplot as plt
import numpy as np

# 1. DEFINE INPUT MATRIX (11 x 5)
matrix_A = np.array([
    [0.5, 0.5, 32, 32, 32],
    [0.5, 0.5, 16, 32, 32],
    [0.5, 30, 16, 16, 16],
    [0.5, 60, 8, 8, 8],
    [0.5, 120, 4, 4, 4],
    [0.5, 160, 2, 2, 2],
    [0.5, 120, 4, 4, 4],
    [0.5, 60, 8, 8, 8],
    [0.5, 30, 16, 16, 16],
    [0.5, 0.5, 16, 32, 32],
    [0.5, 0.5, 32, 32, 8],
], dtype=float)

num_rows, num_cols = matrix_A.shape

# ==========================================
# 2. COMPUTE DYNAMIC TOLERANCE BOUNDARY
# ==========================================
max_entry = np.max(np.abs(matrix_A))
largest_dimension = max(num_rows, num_cols)
machine_epsilon = np.finfo(float).eps  # ~2.22e-16
adaptive_tolerance = largest_dimension * max_entry * machine_epsilon
buffer_tolerance = adaptive_tolerance * 10

# ==========================================
# 3. FULL-MATRIX SVD & UNFORCED REDUCTION
# ==========================================
# Compute full orthogonal bases U (11x11) and Vt (5x5), plus singular values
U, singular_values, Vt = np.linalg.svd(matrix_A, full_matrices=True)

# Unforced 11x5 diagonal matrix via pure matrix multiplication (no hardcoded zeros)
Sigma_unforced = U.T @ matrix_A @ Vt.T

# ==========================================
# 4. NUMERICAL STABILITY, RANK & ENERGY AUDIT
# ==========================================
# Reconstruction accuracy: ||A - U @ Sigma @ Vt||_F
reconstruction_err = np.linalg.norm(
    matrix_A - (U @ Sigma_unforced @ Vt), ord="fro"
)

# Orthogonality verification for U (11x11) and V (5x5)
u_orthogonality_err = np.linalg.norm((U.T @ U) - np.eye(num_rows), ord="fro")
v_orthogonality_err = np.linalg.norm((Vt @ Vt.T) - np.eye(num_cols), ord="fro")

# Unforced entry audit across Sigma_unforced
nonzero_mask = np.abs(Sigma_unforced) > buffer_tolerance
numerical_rank = int(np.sum(singular_values > buffer_tolerance))
nullity = num_cols - numerical_rank

# Condition number (ratio of largest to smallest non-zero singular value)
active_sv = singular_values[singular_values > buffer_tolerance]
condition_number = active_sv[0] / active_sv[-1]

# Variance / Energy capture per singular mode (sigma_i^2 / sum(sigma^2))
total_energy = np.sum(singular_values ** 2)
energy_ratios = (singular_values ** 2) / total_energy * 100.0
cumulative_energy = np.cumsum(energy_ratios)

# ==========================================
# 5. CONSOLE AUDIT REPORT
# ==========================================
print("=== FULL-MATRIX SVD NUMERICAL STABILITY & RANK AUDIT ===")
print(f"Matrix Dimensions                   : {num_rows} × {num_cols}")
print(f"Computed Dynamic Tolerance Boundary : {adaptive_tolerance:.2e}")
print(f"Buffered Audit Threshold (10x)      : {buffer_tolerance:.2e}\n")

print("--- ORTHOGONALITY & BACKWARD STABILITY ---")
print(f"Reconstruction Error ||A - UΣVᵀ||_F : {reconstruction_err:.2e}")
print(f"Left Basis Orthogonality ||UᵀU-I||_F: {u_orthogonality_err:.2e}")
print(f"Right Basis Orthogonality ||VVᵀ-I||_F: {v_orthogonality_err:.2e}")
print(f"Active Condition Number κ(A)        : {condition_number:.2f}\n")

print("--- RANK & SUBSPACE ENERGY BREAKDOWN ---")
print(f"Numerical Rank (Active Modes)       : {numerical_rank} / {num_cols}")
print(f"Column Nullity (Collapsed Modes)    : {nullity} / {num_cols}")
print(f"Total Non-Zero Cells in Σ (> tol)   : {int(np.sum(nonzero_mask))} / {Sigma_unforced.size}\n")

for idx, (sv, pct, cum_pct) in enumerate(
    zip(singular_values, energy_ratios, cumulative_energy), start=1
):
  status = "ACTIVE" if sv > buffer_tolerance else "NULL (Linear Dependency)"
  print(
      f"  Mode σ_{idx}: {sv:10.4f} | Energy: {pct:6.2f}% | "
      f"Cumulative: {cum_pct:6.2f}% | [{status}]"
  )
print("========================================================")

# ==========================================
# 6. MATPLOTLIB UNFORCED DIAGONAL RENDERING
# ==========================================
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.set_xlim(-1.2, num_cols + 0.4)
ax.set_ylim(-0.8, num_rows + 0.8)
ax.invert_yaxis()
ax.axis("off")

ax.text(
    (num_cols - 1) / 2.0,
    -1.05,
    f"Full-Matrix Unforced SVD Diagonal Σ = Uᵀ A V ({num_rows} × {num_cols})\n"
    f"Numerical Rank: {numerical_rank} | Nullity: {nullity} | "
    f"Reconstruction Error: {reconstruction_err:.2e}",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
)

for j in range(num_cols):
  ax.text(
      j, -0.35, f"c{j}", ha="center", va="center",
      fontsize=9, color="#666666", style="italic"
  )

for i in range(num_rows):
  ax.text(
      -0.95, i, f"r{i}", ha="center", va="center",
      fontsize=9, color="#666666", style="italic"
  )
  for j in range(num_cols):
    val = Sigma_unforced[i, j]
    is_active = nonzero_mask[i, j]
    is_diag = i == j

    # Display active singular values cleanly; show exact machine dust elsewhere
    text_str = f"{val:.4g}" if is_active else f"{val:.1e}"

    if is_active:
      color, weight, fsize = "#0b5394", "bold", 11
    elif is_diag:
      color, weight, fsize = "#c5221f", "bold", 9.5  # Highlights collapsed σ_5
    else:
      color, weight, fsize = "#888888", "normal", 9

    ax.text(
        j, i, text_str,
        ha="center", va="center",
        fontsize=fsize, fontfamily="monospace",
        fontweight=weight, color=color,
    )

# Draw square matrix brackets [ ... ]
top_y, bot_y = -0.15, num_rows - 0.85
left_x, right_x, cap = -0.55, num_cols - 0.45, 0.22
ax.plot(
    [left_x + cap, left_x, left_x, left_x + cap],
    [top_y, top_y, bot_y, bot_y],
    color="black", lw=2
)
ax.plot(
    [right_x - cap, right_x, right_x, right_x - cap],
    [top_y, top_y, bot_y, bot_y],
    color="black", lw=2
)

plt.tight_layout()
plt.savefig("full_svd_audit_render.png", dpi=300, bbox_inches="tight")
plt.show()
