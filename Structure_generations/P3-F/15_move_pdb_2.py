import os
import shutil

# Paths
selected_dir = "/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/selected"
source_gro_dir = "/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/MD/individual_frames_gro_MD_local_reset"
target_gro_dir = "/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/lib_gro"

# Ensure target directory exists
os.makedirs(target_gro_dir, exist_ok=True)

# Process each file in selected_total
for filename in os.listdir(selected_dir):
    if not filename.endswith(".pdb"):
        continue

    # Example: frame_1_candidate_433411_AA.pdb
    parts = filename.split('_')
    if len(parts) >= 4:
        frame_id = parts[3]  # '433411'
        gro_filename = f"frame_{frame_id}.gro"
        src_gro_path = os.path.join(source_gro_dir, gro_filename)
        dst_gro_path = os.path.join(target_gro_dir, gro_filename)

        if os.path.exists(src_gro_path):
            shutil.copy(src_gro_path, dst_gro_path)
            print(f"Copied: {gro_filename}")
        else:
            print(f"Missing GRO file for frame: {frame_id}")
