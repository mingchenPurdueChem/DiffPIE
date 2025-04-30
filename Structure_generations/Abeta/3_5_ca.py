#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads .gro files and extracts CA coordinates for residues 1 to 7
Saves the result into a single NumPy array of shape (num_frames, 7, 3)
"""

import os
import numpy as np
from natsort import natsorted

input_dir = '/home/yanbin/Desktop/Projects/GolP/Full_data_code/4_18/Au_F_2/2_Z/merged/group3_output'
target_resids = list(range(2, 9))
ca_atom_name = 'CA'

all_frames_ca_coords = []

def parse_gro_frame(file_path):
    """Parse a .gro file and return a list of (resid, atom_name, xyz) tuples."""
    atoms = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Skip the first two lines: title and atom count
        for line in lines[2:-1]:  # skip box line
            resid = int(line[0:5])
            atom_name = line[10:15].strip()
            x = float(line[20:28]) * 10  # convert from nm to Å
            y = float(line[28:36]) * 10
            z = float(line[36:44]) * 10
            atoms.append((resid, atom_name, np.array([x, y, z])))
    return atoms

for frame_file in natsorted(os.listdir(input_dir)):
    if not frame_file.endswith('.gro'):
        continue

    gro_file = os.path.join(input_dir, frame_file)
    atoms = parse_gro_frame(gro_file)

    frame_coords = []
    for resid in target_resids:
        for a_resid, atom_name, coord in atoms:
            if a_resid == resid and atom_name == ca_atom_name:
                frame_coords.append(coord)
                break
        else:
            print(f"Warning: CA of residue {resid} missing in {frame_file}")
            frame_coords.append(np.full(3, np.nan))  # Use NaNs if missing

    all_frames_ca_coords.append(frame_coords)

ca_all_array = np.array(all_frames_ca_coords)  # shape: (num_frames, 7, 3)
np.save('Ca_residues_1_to_7.npy', ca_all_array)
