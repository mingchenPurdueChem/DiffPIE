import os
import shutil

# Paths
source_root = "data/Str2Str_local_MD_screened"
destination_root = "data/selected"
kept_frames_file = "data/kept_frames.txt"

# Read kept frame names
with open(kept_frames_file, 'r') as f:
    kept_frames = [line.strip() for line in f if line.strip()]

# Ensure output directory exists
os.makedirs(destination_root, exist_ok=True)

# Move matching PDBs
for frame in kept_frames:
    frame_folder = os.path.join(source_root, frame)
    if not os.path.isdir(frame_folder):
        print(f"Warning: {frame_folder} does not exist.")
        continue

    for pdb_file in os.listdir(frame_folder):
        if pdb_file.endswith(".pdb"):
            src = os.path.join(frame_folder, pdb_file)
            dst = os.path.join(destination_root, f"{frame}_{pdb_file}")
            shutil.copy(src, dst)  # use shutil.move(...) if you want to move instead of copy
            print(f"Copied {src} to {dst}")

print("Done moving selected PDBs.")
