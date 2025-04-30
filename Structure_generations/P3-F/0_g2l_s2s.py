#!/usr/bin/env python3
import os
import shutil
import numpy as np

# Load the PDB file
file_path = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/8q1r_pep_ALA.pdb'

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
output_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str'
os.makedirs(output_dir, exist_ok=True)

# Save each frame into its own file
for i, frame in enumerate(frames):
    frame_file_path = os.path.join(output_dir, f'frame_{i+1}.pdb')
    with open(frame_file_path, 'w') as frame_file:
        frame_file.write(''.join(frame))

# Optional: Zip the folder for easier handling
#shutil.make_archive('individual_frames', 'zip', output_dir)

print(f'Split {len(frames)} frames into individual files in "{output_dir}/".')
#print('A zip file "individual_frames.zip" has also been created.')

#####################################
def parse_pdb(file_path):
    """ Parse a .pdb file and return a list of (atom_index, atom_name, xyz, original_line). """
    atoms = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((atom_index, atom_name, np.array([x, y, z]), line))
    return atoms

def build_local_frame(n, ca, c):
    """ Build local coordinate frame using N, Ca, C atoms. """
    x_axis = (n - ca)
    x_axis /= np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis, (c - ca))
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    return ca, np.vstack([x_axis, y_axis, z_axis])

def transform_to_local(coords, origin, axes):
    """ Transform coordinates into local frame defined by origin and axes. """
    return np.dot(coords - origin, axes.T)

def reset_frame_to_local(pdb_file, output_file):
    """ Read PDB, reset all atom coordinates to local frame, and save. """
    atoms = parse_pdb(pdb_file)

    atom_dict = {atom[0]: atom[2] for atom in atoms}

    # These are the atoms that define Ca_res_2: N, Ca, C
    ca_res_2_atoms = [6, 7, 8]

    try:
        n2, ca2, c2 = [atom_dict[idx] for idx in ca_res_2_atoms]
    except KeyError as e:
        print(f"Missing atom {e} in {pdb_file}, skipping.")
        return

    # Build local coordinate system using Ca_res_2
    origin, axes = build_local_frame(n2, ca2, c2)

    # Transform all atoms into the local frame
    transformed_lines = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = int(line[6:11].strip())
                _, _, coords, original_line = [a for a in atoms if a[0] == atom_index][0]
                new_coords = transform_to_local(coords, origin, axes)

                # Rebuild the line with transformed coordinates
                new_line = (
                    original_line[:30] +
                    f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}" +
                    original_line[54:]
                )
                transformed_lines.append(new_line)
            else:
                transformed_lines.append(line)

    # Save new file
    with open(output_file, 'w') as file:
        file.writelines(transformed_lines)

# Paths
input_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str'
output_dir = '/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset'

os.makedirs(output_dir, exist_ok=True)

# Process all PDB files
for frame_file in sorted(os.listdir(input_dir)):
    if not frame_file.endswith('.pdb'):
        continue

    input_file = os.path.join(input_dir, frame_file)
    output_file = os.path.join(output_dir, frame_file)

    reset_frame_to_local(input_file, output_file)

print(f"All frames processed and saved to: {output_dir}")
