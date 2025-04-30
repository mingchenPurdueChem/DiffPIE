#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:16:54 2025

@author: yanbin
"""

import os
from natsort import natsorted

# Define the folders
aa_folder = 'data/individual_frames_pdb_str2str_local_reset_AA_with_H_relaxed/'
sim_base_folder = 'data/selected_pdb_frames_top10'
output_base_folder = 'data/Str2Str_local_MD'

# Make sure output directory exists
os.makedirs(output_base_folder, exist_ok=True)

def average_pdb_lines(line1: str, line2: str) -> str:
    """
    Average coordinates from two PDB ATOM lines and return a new line in line1's format.
    
    Parameters:
        line1 (str): The primary ATOM line (used for formatting output).
        line2 (str): The secondary ATOM line (coordinates to average).
    
    Returns:
        str: A new ATOM line with averaged coordinates.
    """
    x1, y1, z1 = float(line1[30:38]), float(line1[38:46]), float(line1[46:54])
    x2, y2, z2 = float(line2[30:38]), float(line2[38:46]), float(line2[46:54])
    
    x_avg = (x1 + x2) / 2
    y_avg = (y1 + y2) / 2
    z_avg = (z1 + z2) / 2
    
    new_line = f"{line1[:30]}{x_avg:8.3f}{y_avg:8.3f}{z_avg:8.3f}{line1[54:]}"
    return new_line

def replace_pdb_lines(line1: str, line2: str) -> str:
    """
    Average coordinates from two PDB ATOM lines and return a new line in line1's format.
    
    Parameters:
        line1 (str): The primary ATOM line (used for formatting output).
        line2 (str): The secondary ATOM line (coordinates to average).
    
    Returns:
        str: A new ATOM line with averaged coordinates.
    """
    x2, y2, z2 = float(line2[30:38]), float(line2[38:46]), float(line2[46:54])
    
    new_line = f"{line1[:30]}{x2:8.3f}{y2:8.3f}{z2:8.3f}{line1[54:]}"
    return new_line

def update_residue_index_to_zero(lines):
    """
    Update residue index from 1 to 0 in PDB ATOM lines.
    
    Parameters:
        lines (list of str): Lines from the PDB header or atom block.
    
    Returns:
        list of str: Updated lines with residue index changed.
    """
    updated_lines = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Replace columns 22–26 (residue index) with '   0'
            new_line = line[:22] + '   0' + line[26:]
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    return updated_lines

def renumber_atom_indices(pdb_lines):
    """
    Make atom serial numbers continuous for ATOM/HETATM lines.
    
    Parameters:
        pdb_lines (list of str): PDB file lines.

    Returns:
        list of str: PDB lines with continuous atom numbering.
    """
    new_lines = []
    atom_index = 1
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Replace serial index (columns 6–11) with new index
            new_line = f"{line[:6]}{atom_index:5d}{line[11:]}"
            new_lines.append(new_line)
            atom_index += 1
        else:
            new_lines.append(line)
    return new_lines

def fix_and_replace_ter_line(pdb_lines):
    """
    Update TER line serial index to match last ATOM/HETATM + 1.
    """
    # Find last ATOM/HETATM line
    last_atom_line = next(line for line in reversed(pdb_lines) if line.startswith(("ATOM", "HETATM")))
    last_atom_index = int(last_atom_line[6:11])
    new_ter_index = last_atom_index + 1

    # Fix the TER line
    fixed_lines = []
    for line in pdb_lines:
        if line.startswith("TER"):
            # Replace serial number in columns 6–11
            line = f"{line[:6]}{new_ter_index:5d}{line[11:]}"
        fixed_lines.append(line)

    return fixed_lines

def extract_first_protein_model(pdb_lines):
    extracted_lines = []
    inside_model = False
    for line in pdb_lines:
        if line.startswith("MODEL"):
            model_number = int(line.split()[1])
            if model_number > 0 and not inside_model:
                inside_model = True
                continue
            else:
                inside_model = False
        elif line.startswith("ENDMDL"):
            if inside_model:
                break
        elif inside_model:
            extracted_lines.append(line)
    return extracted_lines

for aa_filename in natsorted(os.listdir(aa_folder)):
#for aa_filename in sorted(os.listdir(aa_folder)):
    if not aa_filename.startswith('frame_') or not aa_filename.endswith('.pdb'):
        continue

    frame_index = aa_filename.replace('frame_', '').replace('.pdb', '')

    aa_filepath = os.path.join(aa_folder, aa_filename)
    sim_folder = os.path.join(sim_base_folder, f'frame_{frame_index}')
    output_folder = os.path.join(output_base_folder, f'frame_{frame_index}')

    if not os.path.exists(sim_folder):
        print(f"Skipping {aa_filename} - no folder {sim_folder}")
        continue

    os.makedirs(output_folder, exist_ok=True)

    with open(aa_filepath, 'r') as aa_file:
        aa_lines = aa_file.readlines()

    #aa_lines = extract_first_protein_model(aa_lines)
    aa_lines = [line for line in aa_lines if "REMARK" not in line and "CRYST1" not in line and "MODEL" not in line and "TITLE" not in line]

    for sim_filename in natsorted(os.listdir(sim_folder)):
    #for sim_filename in sorted(os.listdir(sim_folder)):    
        if not sim_filename.endswith('.pdb'):
            continue

        sim_filepath = os.path.join(sim_folder, sim_filename)

        with open(sim_filepath, 'r') as sim_file:
            header_lines = [next(sim_file) for _ in range(21)]

            # Read all the lines from the candidate file (to get lines 22 and 37)
            candidate_lines = sim_file.readlines()
            
            H_lines = candidate_lines[22:41]

            if len(candidate_lines) < 37:
                print(f"Warning: {sim_filepath} has fewer than 37 lines after the header.")
                continue

            # Get the special lines to insert, S and S to replace AA
            line_22 = candidate_lines[8]  # Line 22 is index 21
            line_37 = candidate_lines[19]  # Line 37 is index 36

        #candidate_index = sim_filename.split('_')[1]
        candidate_index = sim_filename.split('_')[-1].split('.')[0]
        output_filepath = os.path.join(output_folder, f'candidate_{candidate_index}_AA.pdb')
        
        header_lines = update_residue_index_to_zero(header_lines)
        H_lines = update_residue_index_to_zero(H_lines)

        combined_lines = header_lines + H_lines + aa_lines  # Start with header + AA lines

        #replace two S from MD
        # combined_lines[58] = average_pdb_lines(combined_lines[58], line_37)
        # combined_lines[146] = average_pdb_lines(combined_lines[146], line_22)
        ###
        combined_lines[53] = replace_pdb_lines(combined_lines[53], line_37)
        combined_lines[141] = replace_pdb_lines(combined_lines[141], line_22)
        
        # remove extra Hs
        #del combined_lines[172]
        #del combined_lines[171]
        #del combined_lines[170]
        #del combined_lines[169]
        #del combined_lines[168]
        #del combined_lines[167]
        #del combined_lines[166]
        del combined_lines[142]
        del combined_lines[54]
        
        combined_lines = renumber_atom_indices(combined_lines)
        combined_lines = fix_and_replace_ter_line(combined_lines)
        
        with open(output_filepath, 'w') as out_file:
            out_file.writelines(combined_lines)

        print(f"Created with insertions: {output_filepath}")

print("All matching pairs processed.")
