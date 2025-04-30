#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:27:47 2025

@author: yanbin
"""

import os
import shutil
import numpy as np

# Load the top 10 closest frame numbers for each frame
top10_closest_frame_numbers = np.load('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/top10_closest_model_indices.npy')
top10_closest_frame_numbers = top10_closest_frame_numbers + 1  # Convert to 1-based indexing if necessary

# Paths
input_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/MD/individual_frames_gro_MD_local_reset'
output_base_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/selected_gro_frames_top10'

# Make sure the top-level output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop over each frame and its 10 candidates
for frame_idx, candidate_frames in enumerate(top10_closest_frame_numbers, start=1):
    # Create a folder for each frame (e.g., frame_1, frame_2, etc.)
    frame_folder = os.path.join(output_base_dir, f'frame_{frame_idx}')
    os.makedirs(frame_folder, exist_ok=True)

    # Copy the 10 closest candidate frames into this folder
    for rank, frame_number in enumerate(candidate_frames, start=1):
        frame_filename = f'frame_{frame_number}.gro'
        src_file = os.path.join(input_dir, frame_filename)

        if os.path.exists(src_file):
            dest_file = os.path.join(frame_folder, f'candidate_{rank}_frame_{frame_number}.gro')
            shutil.copy(src_file, dest_file)
            print(f"Copied: {src_file} -> {dest_file}")
        else:
            print(f"Warning: {src_file} does not exist — skipping.")

print(f"All top 10 candidate frames have been copied into: {output_base_dir}")
