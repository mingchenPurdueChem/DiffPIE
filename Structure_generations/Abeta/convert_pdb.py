#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:40:07 2025

@author: yanbin
"""

import os
from natsort import natsorted
import mdtraj as md

input_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/selected_gro_frames_top10'
output_base_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/selected_pdb_frames_top10'

os.makedirs(output_base_dir, exist_ok=True)

# Loop through each frame_X subfolder
for frame_folder in natsorted(os.listdir(input_dir)):
    frame_folder_path = os.path.join(input_dir, frame_folder)

    if not os.path.isdir(frame_folder_path):
        continue  # Skip if not a folder

    # Create matching subfolder in the output directory
    output_folder_path = os.path.join(output_base_dir, frame_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    # Process each .gro file inside this subfolder
    for gro_file in natsorted(os.listdir(frame_folder_path)):
        if not gro_file.endswith('.gro'):
            continue

        gro_path = os.path.join(frame_folder_path, gro_file)
        pdb_path = os.path.join(output_folder_path, os.path.splitext(gro_file)[0] + '.pdb')

        try:
            traj = md.load(gro_path, top=gro_path)

            # Scale coordinates back to nm (MDTraj expects nm for GRO and Å for PDB)
            traj.xyz *= 0.1  # Convert nm to Å for PDB

            traj.save_pdb(pdb_path)

            print(f"Converted: {gro_path} -> {pdb_path}")
        except Exception as e:
            print(f"Failed to convert {gro_path}: {e}")

print(f"All files processed. PDBs saved in: {output_base_dir}")