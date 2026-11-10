import glob, os, traceback
import numpy as np, pandas as pd

class StructuredVascularTDAEngine:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.output_filename = "ordered_vascular_tda_report.csv"

    def _eval_limit(self, num, denom):
        if np.isclose(num, 0.0) and np.isclose(denom, 0.0): return 1.0
        if np.isclose(denom, 0.0) and not np.isclose(num, 0.0): return np.inf
        return 0.0 if np.isinf(denom) and np.isfinite(num) else num / denom

    def run_ordered_pipeline(self, theta=np.pi / 4, b_limit=1000, a_limit=50):
        if not os.path.isdir(self.directory_path): return print(f"[ERROR] Path not found: {self.directory_path}")
        
        cols = {"case_id", "model_architecture", "total_vessel_length_mm", "bifurcation_count", 
                "mean_tortuosity", "average_branch_length_mm", "fragmentation_errors_h0", "false_loops_h1"}
        
        files = [f for f in glob.glob(os.path.join(self.directory_path, "*.csv")) if self.output_filename not in f]
        pooled = [pd.read_csv(f) for f in files if cols.issubset(pd.read_csv(f, nrows=1).columns)]
        if not pooled: return print("[ABORTED] No valid vascular CSV files found.")
        
        structured_rows = []
        for _, row in pd.concat(pooled, ignore_index=True).iterrows():
            L, H0, H1, O = float(row["total_vessel_length_mm"]), float(row["fragmentation_errors_h0"]), float(row["false_loops_h1"]), float(row["bifurcation_count"])
            tor, br_len = float(row["mean_tortuosity"]), float(row["average_branch_length_mm"])

            # Native Geometrical Decompositions
            p_mag = br_len * tor
            q_mag = br_len / tor if tor != 0 else br_len
            dir_norm = np.sqrt((p_mag * np.cos(theta))**2 + (q_mag * np.sin(theta))**2)
            euc_dist = (L / tor) / (O + 1) if tor != 0 else L
            rho, u_vel, mu = self._eval_limit(L, H0), self._eval_limit(L, H1), self._eval_limit(dir_norm, euc_dist)

            # 1. REYNOLDS TANGENT PLANE CALCULATION FIRST
            reynolds = self._eval_limit((L**3) * euc_dist, H0 * H1 * dir_norm)
            cC = self._eval_limit((L**2) * (b_limit * L) * euc_dist, b_limit * dir_norm)
            cD = self._eval_limit((L**3) * euc_dist, dir_norm * (a_limit * euc_dist))
            cC = 0.0 if np.isinf(cC) or np.isnan(cC) else cC
            cD = 0.0 if np.isinf(cD) or np.isnan(cD) else cD

            # 2. IMPROPER INTEGRALS LOGGED SECOND (DIVERGENT FALLBACK)
            is_div = np.isinf(reynolds) or np.isnan(reynolds)
            if not is_div:
                status, x1, x2, center, integral_val = "CONVERGENT_STEADY_STATE", 0.0, 0.0, 0.0, 0.0
            else:
                status = "DIVERGENT_TDA_SKELETON_TRIGGERED"
                center = tor / L if L > 0 else 1.0
                win = L / np.log1p(abs(cC) + abs(cD)) if (abs(cC) + abs(cD)) > 0 else L
                x1, x2 = float(center - (win / 2.0)), float(center + (win / 2.0))
                
                x_grid = np.linspace(x1, x2, int(np.clip(L, 100, 2000)))
                integral_val = float(np.trapz(cC * (x_grid - center) * np.cos(x_grid / (1.0 + np.log1p(abs(reynolds)))), dx=(x2 - x1)/(len(x_grid)-1)))

            # 3. WOMERSLEY PULSATILE FLOW NUMBER THIRD
            w_sq = self._eval_limit(u_vel * p_mag, mu)
            w_alpha = L * np.sqrt(w_sq) if (w_sq != np.inf and w_sq >= 0) else np.inf

            structured_rows.append({
                "Case_ID": row["case_id"], "Architecture": row["model_architecture"],
                "Reynolds_Number": "DIVERGENT" if np.isinf(reynolds) else round(reynolds, 2),
                "Skeleton_Coeff_C": f"{cC:.4e}", "Skeleton_Coeff_D": f"{cD:.4e}",
                "Integral_Stability_Status": status, "Dynamic_Anchor_Center": round(center, 4),
                "Convergence_Interval_Start": round(x1, 4), "Convergence_Interval_End": round(x2, 4),
                "Evaluated_Fallback_Integral": f"{integral_val:.6e}" if is_div else "N/A",
                "Womersley_Alpha": "DIVERGENT" if np.isinf(w_alpha) else round(w_alpha, 2)
            })

        df = pd.DataFrame(structured_rows)[[
            "Case_ID", "Architecture", "Reynolds_Number", "Skeleton_Coeff_C", "Skeleton_Coeff_D",
            "Integral_Stability_Status", "Dynamic_Anchor_Center", "Convergence_Interval_Start", 
            "Convergence_Interval_End", "Evaluated_Fallback_Integral", "Womersley_Alpha"
        ]]
        df.to_csv(self.output_filename, index=False)
        print(f"[SUCCESS] Report compiled cleanly at: {os.path.abspath(self.output_filename)}")

if __name__ == "__main__":
    StructuredVascularTDAEngine(directory_path="/home/fader/Vessel Project Work").run_ordered_pipeline()

