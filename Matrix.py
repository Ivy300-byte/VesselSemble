import numpy as np
from itertools import combinations
# Type in a 3x3 matrix
matrix_A = np.array([[0.5, 1, 128, 128], [0.5, 64, 128, 128], [0.5, 128, 64, 64], [0.5, 256, 32, 32], [0.5, 512, 16, 16], [0.5, 1024, 16, 16], [0.5, 768, 32, 32], [0.5, 384, 64, 64], [0.5, 192, 128, 128], [0.5, 2, 128, 128]])

num_rows, num_cols = matrix_A.shape

# ==========================================
# 2. COMPUTE DYNAMIC TOLERANCE MEASURE
# ==========================================
# Find the largest structural entry scaling parameter
max_entry = np.max(np.abs(matrix_A))
largest_dimension = max(num_rows, num_cols)

# Calculate the machine precision boundary for standard float64 configurations
machine_epsilon = np.finfo(float).eps  # 2.22e-16

# Adaptive standard calculation
adaptive_tolerance = largest_dimension * max_entry * machine_epsilon

# ==========================================
# 3. RUN LOCALIZED COMBINATORIAL FILTER
# ==========================================
row_combinations = list(combinations(range(num_rows), 2))
col_combinations = list(combinations(range(num_cols), 2))

zero_det_count = 0
right_side_collapse_count = 0
total_submatrices = len(row_combinations) * len(col_combinations)

for r_indices in row_combinations:
    for c_indices in col_combinations:
        # Isolate the specific 2x2 coordinate patch matrix natively
        sub_matrix = matrix_A[np.ix_(list(r_indices), list(c_indices))]
        
        # Calculate local determinant
        det_val = np.linalg.det(sub_matrix)
        
        # Check against adaptive threshold boundaries
        if np.abs(det_val) < adaptive_tolerance:
            zero_det_count += 1
            
            # Check for localized structural zero alignments on the right-hand boundary
            if np.all(np.abs(sub_matrix[:, -1]) < adaptive_tolerance):
                right_side_collapse_count += 1

# ==========================================
# REPORT ADJUSTED FOR SYSTEM PRECISION
# ==========================================
print("=== DYNAMIC SUBSPACE ACCURACY REPORT ===")
print(f"Computed Dynamic Tolerance Boundary : {adaptive_tolerance:.2e}")
print(f"Total 2x2 Coordinate Patches Tested : {total_submatrices}")
print(f"Micro-spaces Collapsed (Det < Bound): {zero_det_count} / {total_submatrices}")
print(f"Right-side structural flatlines      : {right_side_collapse_count}")
print("========================================")
