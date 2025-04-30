#!/usr/bin/env python3
import os
import shutil
import numpy as np

# Load the PDB file
file_path = 'data/KLVFFAE.pdb'

with open(file_path, 'r') as file:
    pdb_lines = file.readlines()

# Split frames using ENDMDL as separator
frames = []
current_frame = []

for line in pdb_lines:
    current_frame.append(line)
    if line.startswith('ENDMDL'):
        frames.append(current_frame)
        current_frame = []

# Ensure output directory exists
output_dir = 'data/individual'
os.makedirs(output_dir, exist_ok=True)

# Save each frame into its own file
for i, frame in enumerate(frames):
    frame_file_path = os.path.join(output_dir, f'KLVFFAE_2_{i+1}.pdb')
    with open(frame_file_path, 'w') as frame_file:
        frame_file.write(''.join(frame))

# Optional: Zip the folder for easier handling
#shutil.make_archive('individual_frames', 'zip', output_dir)

print(f'Split {len(frames)} frames into individual files in "{output_dir}/".')
#print('A zip file "individual_frames.zip" has also been created.')

#####################################
