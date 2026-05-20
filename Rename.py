import os

def sync_by_time(ref_dir, target_dir):
    # Sort files by the time they were created/modified
    ref_files = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.nii.gz')], key=os.path.getmtime)
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.nii.gz')], key=os.path.getmtime)

    if len(ref_files) != len(target_files):
        print("Warning: Folder sizes don't match. Chronological matching might be inaccurate!")

    for r_path, t_path in zip(ref_files, target_files):
        new_name = os.path.basename(r_path)
        old_path = t_path
        new_path = os.path.join(target_dir, new_name)
        
        print(f"[TIME MATCH] {os.path.basename(t_path)} -> {new_name}")
        os.rename(old_path, new_path)

sync_by_time("/home/fader/Task08_HepaticVessel/Task08_HepaticVessel/imagesTs", "/home/fader/predictions")

