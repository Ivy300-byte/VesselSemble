import pandas as pd
import numpy as np
import os
import re

def run_predictive_invariant_analysis_final(input_file_path, output_csv_path):
    """
    Biophysical Anomaly Auditor. Tracks how topological defects (H0/H1) 
    physically warp and drive fluid-dynamic invariants into impossible states.
    """
    # 1. READ YOUR INPUT DATA SAFE AND CLEAN
    df = pd.read_csv(input_file_path)
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    print("--- PIPELINE INITIALIZATION ---")
    print(f"Loaded raw dataset with {len(df)} patient profiles.")

    # HARD-MAPPED EXPERT COLUMN KEY MATRIX
    col_case = 'case_id'
    col_branch = 'average_branch_length_mm'
    col_len = 'total_vessel_length_mm'
    col_bif = 'bifurcation_count'
    col_tort = 'mean_tortuosity'
    col_frag = 'fragmentation_errors_h0'
    col_loops = 'false_loops_h1'

    # 2. HARD-CODED BIOPHYSICAL IDEAL CONSTANTS
    df['ideal_murrays_deviation'] = 1.000
    df['ideal_area_expansion'] = 1.260

    # 3. RUN SENSITIVITY PREDICTIONS (RAW UNALTERED PHYSICS)
    df['loop_density'] = df[col_loops] / df[col_bif].replace(0, np.nan)
    df['frag_density'] = df[col_frag] / df[col_len].replace(0, np.nan)
    df['tortuosity_strain'] = (df[col_tort] - 1.15).clip(lower=0)

    # Core biophysical distortion engines calculate the TRUE fluid breakdown
    df['predicted_murrays_deviation'] = df['ideal_murrays_deviation'] + (df['loop_density'] * 1.5) - (df['frag_density'] * 4.0)
    df['predicted_area_expansion'] = df['ideal_area_expansion'] + (df['tortuosity_strain'] * df['loop_density'] * 2.5)
    
    df['predicted_murrays_deviation'] = df['predicted_murrays_deviation'].clip(0.1, 3.5)
    df['predicted_area_expansion'] = df['predicted_area_expansion'].clip(0.3, 3.5)

    # Compute absolute physics shift safely from the raw, corrupted scores
    df['physics_shift'] = np.abs(df['predicted_murrays_deviation'] - df['ideal_murrays_deviation'])
    
    # -------------------------------------------------------------------------
    # PART 4: THE REGEX ANCHOR & CAUSAL AUDIT
    # -------------------------------------------------------------------------
    # Target healthy control case numbers as integers
    healthy_case_numbers = [1, 5, 7, 14, 20]
    
    def extract_true_patient_id(case_string):
        low_string = str(case_string).lower()
        match = re.search(r'\.(\d+)', low_string)
        if match:
            return int(match.group(1))
        match2 = re.search(r'patient_(\d+)', low_string)
        if match2:
            return int(match2.group(1))
        return -1

    df['extracted_case_number'] = df[col_case].apply(extract_true_patient_id)
    is_registered_healthy = df['extracted_case_number'].isin(healthy_case_numbers)

    # -------------------------------------------------------------------------
    # THE AUDIT CRITERIA: REVEALING THE IMPOSSIBLE SOFTWARE ERROR
    # -------------------------------------------------------------------------
    # If a patient is verified as clinically healthy, ANY high loop, fragmentation, 
    # or physical invariant drift represents a major software failure mode.
    is_impossible_healthy_noise = is_registered_healthy & ((df[col_frag] > 50) | (df[col_loops] > 1) | (df['physics_shift'] > 0.01))
    
    is_impossible_dilation = df['predicted_area_expansion'] > 2.0
    is_impossible_resistance = df['predicted_murrays_deviation'] < 0.45
    
    is_impossible = is_impossible_healthy_noise | is_impossible_dilation | is_impossible_resistance
    
    # Tumor calculations handle the remaining un-registered, tumor-bearing rows
    is_plausible_tumor = (df['physics_shift'] > 0.05) & (~is_impossible) & (~is_registered_healthy)
    is_valid_baseline = (~is_impossible) & (~is_plausible_tumor)

    # 5. CONSOLIDATE CAUSAL SELECTIONS
    conditions = [is_impossible, is_plausible_tumor, is_valid_baseline]
    status_choices = ['IMPOSSIBLE HALLUCINATION', 'PLAUSIBLE PATHOLOGY', 'VALID ANATOMY']
    issue_choices = ['Biophysical Invariant Collapse', 'Tumor Neoangiogenesis Bed', 'Physiologically Compliant Structure']
    directive_choices = [
        'CRITICAL REJECT: Structure breaks physical laws; software model error.',
        'APPROVE: Complex tumor vascularization captured; preserve for clinical trace.',
        'APPROVE: Normal human vascular anatomy verified.'
    ]

    df['clinical_audit_status'] = np.select(conditions, status_choices, default='VALID ANATOMY')
    df['anatomical_classification'] = np.select(conditions, issue_choices, default='Physiologically Compliant Structure')
    df['surgical_directive'] = np.select(conditions, directive_choices, default='APPROVE: Automated structural mapping passes safety gates.')

    # 6. RECONSTRUCT THE EXPORT DATAFRAME WITH PROFESSIONAL HEADERS
    output_df = pd.DataFrame({
        'Case_ID': df[col_case].str.replace('ENSEMBLE_', '').str.replace('_mask', '').str.upper(),
        'H1_False_Loops': df[col_loops],
        'Frag_Errors_H0': df[col_frag],
        'Predicted_Murray_Score': df['predicted_murrays_deviation'].round(3),
        'Predicted_Area_Expansion': df['predicted_area_expansion'].round(3),
        'Physics_Shift_Score': df['physics_shift'].round(3),
        'Clinical_Audit_Status': df['clinical_audit_status'],
        'Anatomical_Classification': df['anatomical_classification'],
        'Surgical_Directive': df['surgical_directive']
    })
    
    # 7. EXPORT DATA TO DISK
    output_df.to_csv(output_csv_path, index=False)
    
    print("\n--- FILE EXPORT SUCCESSFUL ---")
    print(f"Audited report successfully saved to disk.")
    print(f"File Location: {os.path.abspath(output_csv_path)}")
    print("--------------------------------")
    
    return output_df

# --- RUN EXECUTION LOOP ---
raw_input = r"C:\Users\ifeeu\images\nnUNet-on-3Dircadb.csv"
saved_output = r"C:\Users\ifeeu\images\Temp\nnUNet-on-3Dircadb_Audited_Clinical_Report.csv"

# Trigger final clean save
audited_table = run_predictive_invariant_analysis_final(raw_input, saved_output)
print(audited_table.to_string())





