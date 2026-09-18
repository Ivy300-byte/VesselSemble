import os
import re

def batch_rename_fix(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".nii.gz")]
    
    for filename in files:
        # This regex looks for the number in the original name
        # e.g., in 'prediction_7.nii.gz', it finds '7'
        match = re.search(r'(\d+)', filename)
        
        if match:
            patient_id = match.group(1)
            # Create the specific 3Dircadb1.X format
            new_name = f"3Dircadb1.{patient_id}.nii.gz"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # Prevent overwriting if the name is already correct
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Fixed: {filename} -> {new_name}")
        else:
            print(f"Could not find a number in: {filename}")

# Run for both directories
batch_rename_fix("/home/fader/predictions")



