#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:12:23 2025

@author: yanbin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:12:23 2025

@author: yanbin
"""

import os
import numpy as np
from natsort import natsorted

# Folder containing processed PDB files (local reset frames)
input_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset_AA_with_H_relaxed/'

# Residue number and atom name for CA in residue 10
ca_res_10_resid = 9
ca_atom_name = 'CA'

# Storage for the coordinates of CA atom in residue 10
ca10_local_coords = []

def parse_pdb_frame(file_path):
    """ Parse a .pdb file and return a list of (resid, atom_name, xyz) tuples. """
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                res_id = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((res_id, atom_name, np.array([x, y, z])))
    return atoms

# Process each .pdb file in the directory
for frame_file in natsorted(os.listdir(input_dir)):
    if not frame_file.endswith('.pdb'):
        continue

    pdb_file = os.path.join(input_dir, frame_file)

    # Parse atoms from the pdb file
    atoms = parse_pdb_frame(pdb_file)

    # Find CA atom in residue 10
    for resid, atom_name, coord in atoms:
        if resid == ca_res_10_resid and atom_name == ca_atom_name:
            ca10_local_coords.append(coord)
            break
    else:
        print(f"Warning: CA of residue 10 missing in {frame_file}")

# Convert to numpy array
ca10_local_coords_array = np.array(ca10_local_coords)
#print(ca10_local_coords_array[0, 2])
#print(ca10_local_coords_array[38, 2])

# Save to file
np.save('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/Ca10_local_coordinates_str2str.npy', ca10_local_coords_array)

