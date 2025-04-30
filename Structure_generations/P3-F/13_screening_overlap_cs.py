import os
import shutil
import numpy as np
from itertools import combinations
from natsort import natsorted

# Parameters
clash_distance_threshold = 0.9  # in nm
c_s_distance_threshold = 2.0    # in nm
s_index = 137  # 0-based
c_index = 134  # cb

# Paths
input_base_folder = 'data/Str2Str_local_MD'
output_base_folder = 'data/Str2Str_local_MD_screened'
kept_list_file = 'data/kept_structures.txt'

def parse_coords(pdb_file):
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def has_atom_overlap(coords, threshold=clash_distance_threshold):
    for i, j in combinations(range(len(coords)), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < threshold:
            return True
    return False

def is_c_s_distance_valid(coords, c_index, s_index, threshold):
    if c_index >= len(coords) or s_index >= len(coords):
        return False
    dist = np.linalg.norm(coords[c_index] - coords[s_index])
    return dist <= threshold

# Open the kept list file
with open(kept_list_file, 'w') as kept_file:
    for frame_folder in natsorted(os.listdir(input_base_folder)):
        frame_folder_path = os.path.join(input_base_folder, frame_folder)

        if not os.path.isdir(frame_folder_path):
            continue

        output_frame_folder = os.path.join(output_base_folder, frame_folder)
        os.makedirs(output_frame_folder, exist_ok=True)

        for candidate in sorted(os.listdir(frame_folder_path)):
            if not candidate.endswith('.pdb'):
                continue

            candidate_path = os.path.join(frame_folder_path, candidate)
            coords = parse_coords(candidate_path)

            if has_atom_overlap(coords):
                #print(f"Skip {candidate} (clash detected)")
                continue

            if not is_c_s_distance_valid(coords, c_index, s_index, c_s_distance_threshold):
                #print(f"Skip {candidate} (C-S distance too large)")
                continue

            # Passed all filters
            shutil.copy(candidate_path, os.path.join(output_frame_folder, candidate))
            print(f"Kept {candidate} (no clash, valid C-S distance)")
            kept_file.write(f"{frame_folder}/{candidate}\n")

print("Screening complete.")
print(f"Screened files saved in: {output_base_folder}")
print(f"Kept structure list saved in: {kept_list_file}")

kept_file = "data/kept_structures.txt"
output_file = "data/kept_frames.txt"
frame_labels = set()

# Read and collect unique frame labels
with open(kept_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("frame_"):
            frame = line.split('/')[0]  # e.g., 'frame_4'
            frame_labels.add(frame)

# Sort and write to output
with open(output_file, 'w') as out:
    for frame in sorted(frame_labels):
        out.write(frame + '\n')

print(f"Saved kept frames to: {output_file}")

from natsort import natsorted

kept_frames_file = 'data/kept_frames.txt'

# Read frame labels
with open(kept_frames_file, 'r') as f:
    frames = [line.strip() for line in f if line.strip()]

# Natural sort
sorted_frames = natsorted(frames)

# Overwrite the file with sorted frames
with open(kept_frames_file, 'w') as f:
    for frame in sorted_frames:
        f.write(frame + '\n')

print("Kept frames sorted and saved back to kept_frames.txt")
