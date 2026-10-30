import numpy as np
from itertools import combinations

# Type in the matrix
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
])

num_rows, num_cols = matrix_A.shape
# ==========================================
# 2. COMPUTE DYNAMIC TOLERANCE MEASURE
# ==========================================
max_entry = np.max(np.abs(matrix_A))
largest_dimension = max(num_rows, num_cols)
machine_epsilon = np.finfo(float).eps # 2.22e-16
adaptive_tolerance = largest_dimension * max_entry * machine_epsilon
buffer_tolerance = adaptive_tolerance * 10

# ==========================================
# 3. RUN LOCALIZED COMBINATORIAL FILTER
# ==========================================
row_combinations = list(combinations(range(num_rows), 2))
col_combinations = list(combinations(range(num_cols), 2))
total_submatrices = len(row_combinations) * len(col_combinations)
identity_2x2 = np.eye(2)

# TRACK 1: COLLAPSED GEOMETRY (Det ≈ 0)
zero_det_count = 0
c_verified_eigenvalue_count = 0
c_repeated_eigenvalue_count = 0
c_complex_eigenvalue_count = 0

# TRACK 2: STABLE GEOMETRY (Det ≠ 0)
stable_det_count = 0
s_verified_eigenvalue_count = 0
s_repeated_eigenvalue_count = 0
s_complex_eigenvalue_count = 0

# NEW METRIC: ANOMALY TRACKING REGISTER
numerical_anomalies_count = 0
anomaly_coordinates = []

for r_indices in row_combinations:
    for c_indices in col_combinations:
        sub_matrix = matrix_A[np.ix_(list(r_indices), list(c_indices))]
        det_val = np.linalg.det(sub_matrix)
        
        # Calculate local eigenvalues/vectors
        evals, evecs = np.linalg.eig(sub_matrix)
        
        # Characteristic Equation Check: det(A - lambda * I) ≈ 0
        char_eq_verified = False
        for lam in evals:
            characteristic_matrix = sub_matrix - lam * identity_2x2
            char_det = np.linalg.det(characteristic_matrix)
            if np.abs(char_det) < buffer_tolerance:
                char_eq_verified = True

        # BRANCH A: COLLAPSED TRACK
        if np.abs(det_val) < buffer_tolerance:
            zero_det_count += 1
            if char_eq_verified:
                c_verified_eigenvalue_count += 1
            if np.abs(evals[0] - evals[1]) < buffer_tolerance:
                c_repeated_eigenvalue_count += 1
            if np.any(np.abs(np.imag(evals)) > buffer_tolerance):
                c_complex_eigenvalue_count += 1
                
        # BRANCH B: STABLE TRACK
        else:
            stable_det_count += 1
            if char_eq_verified:
                s_verified_eigenvalue_count += 1
                if np.abs(evals[0] - evals[1]) < buffer_tolerance:
                    s_repeated_eigenvalue_count += 1
                if np.any(np.abs(np.imag(evals)) > buffer_tolerance):
                    s_complex_eigenvalue_count += 1
            else:
                # Catching the unverified spaces that leak through Stage 1 but fail Stage 2
                numerical_anomalies_count += 1
                anomaly_coordinates.append((r_indices, c_indices, sub_matrix))

# ==========================================
# REPORT ADJUSTED FOR TRACK SEPARATION & ANOMALIES
# ==========================================
denom_c = zero_det_count if zero_det_count > 0 else 1
denom_s = stable_det_count if stable_det_count > 0 else 1
denom_verified_s = s_verified_eigenvalue_count if s_verified_eigenvalue_count > 0 else 1

print("=== ADVANCED SUBSPACE GEOMETRY & ANOMALY REPORT ===")
print(f"Computed Dynamic Tolerance Boundary  : {adaptive_tolerance:.2e}")
print(f"Total 2x2 Coordinate Patches Tested  : {total_submatrices}\n")

print("--- TRACK 1: COLLAPSED GEOMETRY BLOCK (Det ≈ 0) ---")
print(f"Stage 1 Pool Size (Det ≈ 0)          : {zero_det_count} / {total_submatrices}")
print(f"Stage 2 Verified (det(A-λI)≈0)       : {c_verified_eigenvalue_count} / {denom_c}")
print(f"Patches with Repeated Eigenvalues    : {c_repeated_eigenvalue_count} / {denom_c}")
print(f"Patches with Complex Eigenvalues     : {c_complex_eigenvalue_count} / {denom_c}\n")

print("--- TRACK 2: STABLE GEOMETRY BLOCK (Det ≠ 0) ---")
print(f"Stage 1 Pool Size (Det ≠ 0)          : {stable_det_count} / {total_submatrices}")
print(f"Stage 2 Verified (det(A-λI)≈0)       : {s_verified_eigenvalue_count} / {denom_s}")
print(f"Patches with Repeated Eigenvalues    : {s_repeated_eigenvalue_count} / {denom_verified_s}")
print(f"Patches with Complex Eigenvalues     : {s_complex_eigenvalue_count} / {denom_verified_s}\n")

print("--- NEW METRIC: SYSTEM ANOMALIES (Stage 1 Independent but Stage 2 Unclean) ---")
print(f"Total Unclean Numerical Anomalies    : {numerical_anomalies_count} / {stable_det_count}")
print("-------------------------------------------------------------------------------")
if numerical_anomalies_count > 0:
    print("Detected Anomaly Submatrix Patch Profiles:")
    for idx, (r_idx, c_idx, sub_mat) in enumerate(anomaly_coordinates):
        print(f"\n[Anomaly #{idx+1}] Rows: {r_idx} | Columns: {c_idx}")
        print(sub_mat)
print("===============================================================================")

