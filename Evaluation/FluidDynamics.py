import glob
import os
import traceback
import numpy as np
import pandas as pd


class RealDataTDABoundaryEngine:

    def __init__(self, directory_path=None):
        """Initializes the framework with an empty DataFrame or loads a target directory."""
        self.df = pd.DataFrame()
        if directory_path:
            self.import_from_directory(directory_path)

    def import_from_directory(self, directory_path: str):
        """
        Scans a target directory, looks for CSV files, filters them for expected 
        vascular columns, and aggregates them seamlessly into a unified pipeline.
        """
        print(f"\n[DEBUG INIT] Checking if target path exists: {directory_path}")
        if not os.path.isdir(directory_path):
            print(f"[CRITICAL ERROR] Directory does not exist: {directory_path}")
            raise FileNotFoundError(f"The specified directory was not found: {directory_path}")
            
        print(f"[DIRECTORY SCAN] Searching for CSV files inside: {os.path.abspath(directory_path)}")
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        print(f"[DIRECTORY SCAN] Found {len(csv_files)} total CSV file candidates.")
        
        expected_columns = {
            "case_id", "model_architecture", "total_vessel_length_mm", 
            "bifurcation_count", "mean_tortuosity", "average_branch_length_mm", 
            "fragmentation_errors_h0", "false_loops_h1"
        }
        
        dataframes_to_merge = []
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            print(f"\n[PARSING FILE] Inspecting: {file_name}")
            try:
                # Read columns to verify template matches
                preview_df = pd.read_csv(file_path, nrows=1)
                found_cols = set(preview_df.columns)
                missing_cols = expected_columns - found_cols
                
                if not missing_cols:
                    full_df = pd.read_csv(file_path)
                    print(f" [SUCCESS] Matches requirements. Ingested {len(full_df)} records from {file_name}")
                    dataframes_to_merge.append(full_df)
                else:
                    print(f" [SKIPPED] Structure mismatch. Missing required columns: {missing_cols}")
            except Exception as e:
                print(f" [PARSING FAILED] Internal reading error on {file_name}: {str(e)}")
                print(traceback.format_exc())
                
        if dataframes_to_merge:
            self.df = pd.concat(dataframes_to_merge, ignore_index=True)
            print(f"\n[INGESTION COMPLETE] Total aggregated active working dataset size: {len(self.df)} records.")
        else:
            print("\n[INGESTION WARNING] No valid files matched the schema. self.df is empty.")

    def evaluate_indeterminate_limit(self, row_id, var_name, numerator, denominator):
        """
        Analytically monitors continuous limits of indeterminate and divergent forms
        (#/0, 0/0, #/inf) mimicking standard L'Hôpital rule evaluations.
        """
        if np.isclose(numerator, 0.0) and np.isclose(denominator, 0.0):
            print(f"   [L'HOPITAL TRIGGERED] Form 0/0 detected for {var_name} on {row_id}. Normalizing limit proxy to 1.0")
            return 1.0 

        if np.isclose(denominator, 0.0) and not np.isclose(numerator, 0.0):
            print(f"   [DIVERGENCE DETECTED] Form #/0 (Infinity) found for {var_name} on {row_id}.")
            return np.inf

        if np.isinf(denominator) and np.isfinite(numerator):
            print(f"   [LIMIT ZERO DETECTED] Form #/inf found for {var_name} on {row_id}. Converges to 0.0")
            return 0.0

        return numerator / denominator

    def compute_skeleton_and_convergence(self, row, theta=np.pi / 4, b_limit=1000, a_limit=50):
        """
        Builds the non-singular structural skeleton of the Reynolds Tangent Plane
        and maps out the exact local coordinate profile of integral convergence.
        """
        case_id = row["case_id"]
        L = float(row["total_vessel_length_mm"])
        H0 = float(row["fragmentation_errors_h0"])
        H1 = float(row["false_loops_h1"])
        O = float(row["bifurcation_count"])
        tortuosity = float(row["mean_tortuosity"])
        branch_length = float(row["average_branch_length_mm"])

        # Dynamic Parametric Vector Adjustments
        p_mag = branch_length * tortuosity
        q_mag = branch_length / tortuosity if tortuosity != 0 else branch_length
        dir_norm = np.sqrt((p_mag * np.cos(theta)) ** 2 + (q_mag * np.sin(theta)) ** 2)
        euclidean_distance = (L / tortuosity) / (O + 1) if tortuosity != 0 else L

        # Calculate properties using deep-trace error protection
        rho = self.evaluate_indeterminate_limit(case_id, "rho_density", L, H0)
        u = self.evaluate_indeterminate_limit(case_id, "u_velocity", L, H1)
        mu = self.evaluate_indeterminate_limit(case_id, "mu_viscosity", dir_norm, euclidean_distance)

        # Circling expression from Page 11: Re = (L^3 * Euc) / (H0 * H1 * dir_norm)
        denom_re = H0 * H1 * dir_norm
        reynolds = self.evaluate_indeterminate_limit(case_id, "reynolds_number", (L**3) * euclidean_distance, denom_re)

        # Page 19 Tangent Plane Skeleton Coefficients
        num_C = (L**2) * (b_limit * L) * euclidean_distance
        denom_C = b_limit * dir_norm
        coeff_C = self.evaluate_indeterminate_limit(case_id, "plane_coeff_C", num_C, denom_C)

        num_D = (L**3) * euclidean_distance
        denom_D = dir_norm * (a_limit * euclidean_distance)
        coeff_D = self.evaluate_indeterminate_limit(case_id, "plane_coeff_D", num_D, denom_D)

        # Fallback values if absolute infinity issues threaten computational bounds
        coeff_C = 0.0 if np.isinf(coeff_C) or np.isnan(coeff_C) else coeff_C
        coeff_D = 0.0 if np.isinf(coeff_D) or np.isnan(coeff_D) else coeff_D

        # Coordinate Spatial Search Bounds Zone Optimization (Page 19 integration anchor check)
        x0, y0 = 0.0, 0.0
        x_space = np.linspace(-100, 100, 200)
        y_space = np.linspace(-100, 100, 200)
        
        convergent_x, convergent_y = x0, y0
        min_dissipation_energy = np.inf

        for x in x_space:
            for y in y_space:
                z_value = coeff_C * (x - x0) + coeff_D * (y - y0)
                if np.isfinite(z_value):
                    energy = abs(z_value)
                    if energy < min_dissipation_energy:
                        min_dissipation_energy = energy
                        convergent_x = x
                        convergent_y = y

        return {
            "reynolds": reynolds,
            "C": coeff_C,
            "D": coeff_D,
            "conv_x": convergent_x,
            "conv_y": convergent_y
        }

    def pipeline_run(self):
        """Processes rows dynamically, generating explicit prints and automated file checkouts."""
        if self.df.empty:
            print("\n[PIPELINE ABORTED] The active memory dataset is empty. Check path logs above.")
            return pd.DataFrame()
            
        print(f"\n[PIPELINE START] Beginning analytical processing on {len(self.df)} lines...")
        output_metrics = []
        
        for idx, row in self.df.iterrows():
            print(f" -> Calculating constraints for case entry index #{idx}: {row['case_id']}")
            res = self.compute_skeleton_and_convergence(row)
            
            output_metrics.append({
                "case_id": row["case_id"],
                "architecture": row["model_architecture"],
                "reynolds_form": "DIVERGENT (inf)" if np.isinf(res["reynolds"]) else round(res["reynolds"], 2),
                "skeleton_C": f"{res['C']:.2e}",
                "skeleton_D": f"{res['D']:.2e}",
                "convergence_point_x": round(res["conv_x"], 4),
                "convergence_point_y": round(res["conv_y"], 4)
            })
            
        result_df = pd.DataFrame(output_metrics)
        
        # --- AUTOMATED SECURITY CHECKOUT OVERWRITES ---
        debug_output_filename = "tda_debug_pipeline_output.csv"
        try:
            result_df.to_csv(debug_output_filename, index=False)
            print(f"\n[DEBUG FILE EXPORTED] Successfully saved running snapshot to: {os.path.abspath(debug_output_filename)}")
        except Exception as export_error:
            print(f"\n[EXPORT ERROR] Failed to save copy to disk space: {str(export_error)}")
            
        return result_df


# ==========================================
# DIAGNOSTIC EXECUTION RUNNER
# ==========================================
if __name__ == "__main__":
    # Change this path to the location containing your actual CSV documents
    target_directory_path = "/home/fader/Vessel Project Work"
    
    # Run analysis engine directly via folder lookup
    try:
        engine = RealDataTDABoundaryEngine(directory_path=target_directory_path)
        analysis_report = engine.pipeline_run()
        
        if not analysis_report.empty:
            print("\n" + "="*50)
            print("FINAL CONSOLE SUMMARY METRICS REPORT:")
            print("="*50)
            print(analysis_report.to_string(index=False))
    except FileNotFoundError as fnf_error:
        print(f"\n[CONFIGURATION NEEDED] Please create the folder or update target_directory_path: {str(fnf_error)}")

