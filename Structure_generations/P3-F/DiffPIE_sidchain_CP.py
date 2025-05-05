import os
import shutil
import numpy as np
import glob
import mdtraj as md
import re
from natsort import natsorted
from openmm.app import *
from openmm import *
from openmm.unit import *
from io import StringIO
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from scipy.spatial import cKDTree
from itertools import combinations
#####################################
str2str_path = '/Str2Str/logs/inference/runs/cp_ori_1e4/samples/all_delta/'# Load the PDB file
str2str_pdb_list = glob.glob(os.path.join(str2str_path, '*.pdb'))
str2str_pdb = str2str_pdb_list[0]
base_path = './Data'
os.makedirs(base_path, exist_ok=True)
MD_path = './MD'
os.makedirs(MD_path, exist_ok=True)
ind_gol_dir = f"{base_path}/individual_frames_pdb_str2str"
os.makedirs(ind_gol_dir, exist_ok=True)
ind_loc_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset"
os.makedirs(ind_loc_dir, exist_ok=True)
aa_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_AA"
os.makedirs(aa_dir, exist_ok=True)
ace_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_ACE"
os.makedirs(ace_dir, exist_ok=True)
aceh_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_ACE_with_H"
os.makedirs(aceh_dir, exist_ok=True)
aceh_125_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_ACE_with_H/atoms_125" # this is sepcical case as sometime it form c-s
os.makedirs(aceh_125_dir, exist_ok=True)
aceh_125_reset_dir = f"{base_path}/reset_index"
os.makedirs(aceh_125_reset_dir, exist_ok=True)
aceh_125_relax_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_ACE_with_H_relaxed"
os.makedirs(aceh_125_relax_dir, exist_ok=True)
aceh_125_local_relax_dir = f"{base_path}/individual_frames_pdb_str2str_local_reset_AA_with_H_relaxed"
os.makedirs(aceh_125_local_relax_dir, exist_ok=True)
energy_file = f"{base_path}/peptide_energy.txt"
output_base_gro_dir = f'{base_path}/selected_gro_frames_top10'
os.makedirs(output_base_gro_dir, exist_ok=True) # Make sure the top-level output directory exists
output_base_pdb_dir = f'{base_path}/selected_pdb_frames_top10'
os.makedirs(output_base_pdb_dir, exist_ok=True)
assemble_MD = f'{base_path}/Str2Str_local_MD'
os.makedirs(assemble_MD, exist_ok=True)
screen_assemble_MD = f'{base_path}/Str2Str_local_MD_screened'
os.makedirs(screen_assemble_MD, exist_ok=True)
destination_root = f'{base_path}/selected'
os.makedirs(destination_root, exist_ok=True)
MD_lib = f'{base_path}/lib_gro'
os.makedirs(MD_lib, exist_ok=True)
#####################################
#####################################
with open(str2str_pdb, 'r') as file:
    pdb_lines = file.readlines()
frames = [] # Split frames using ENDMDL as separator
current_frame = []
for line in pdb_lines:
    current_frame.append(line)
    if line.startswith('ENDMDL'):
        frames.append(current_frame)
        current_frame = []
for i, frame in enumerate(frames): # Save each frame into its own file
    frame_file_path = os.path.join(ind_gol_dir, f'frame_{i+1}.pdb')
    with open(frame_file_path, 'w') as frame_file:
        frame_file.write(''.join(frame))
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
    ca_res_2_atoms = [6, 7, 8] # These are the atoms that define Ca_res_2: N, Ca, C
    try:
        n2, ca2, c2 = [atom_dict[idx] for idx in ca_res_2_atoms]
    except KeyError as e:
        print(f"Missing atom {e} in {pdb_file}, skipping.")
        return
    origin, axes = build_local_frame(n2, ca2, c2) # Build local coordinate system using Ca_res_2

    transformed_lines = [] # Transform all atoms into the local frame
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
    with open(output_file, 'w') as file:# Save new file
        file.writelines(transformed_lines)

# Process all PDB files
for frame_file in sorted(os.listdir(ind_gol_dir)):
    if not frame_file.endswith('.pdb'):
        continue
    input_file = os.path.join(ind_gol_dir, frame_file)
    output_file = os.path.join(ind_loc_dir, frame_file)
    reset_frame_to_local(input_file, output_file)
#####################################
# Loop through all PDB files and run FASPR
for pdb_file in glob.glob(os.path.join(ind_loc_dir, '*.pdb')):
    base_name = os.path.basename(pdb_file)
    output_name = os.path.join(aa_dir, f"{os.path.splitext(base_name)[0]}_AA.pdb")
    subprocess.run(['./FASPR', '-i', pdb_file, '-o', output_name])
def convert_ala_to_ace(input_pdb_path, output_pdb_path):
    with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()

                if res_name == "ALA":
                    if atom_name in {"N", "CB"}:
                        continue  # Remove unwanted atoms
                    elif atom_name == "CA":
                        # Change atom name to CH3 and residue to ACE
                        line = line[:12] + " CH3" + line[16:]
                        line = line[:17] + "ACE" + line[20:]
                    elif atom_name in {"C", "O"}:
                        # Change only residue name to ACE
                        line = line[:17] + "ACE" + line[20:]
                outfile.write(line)
            else:
                outfile.write(line)

def process_all_pdbs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_ala_to_ace(input_path, output_path)

process_all_pdbs(aa_dir, ace_dir)
#####################################
for pdb_file in glob.glob(os.path.join(ace_dir, '*.pdb')):
    base_name = os.path.basename(pdb_file)
    output_name = os.path.join(aceh_dir, f"{os.path.splitext(base_name)[0]}_H.pdb")
    subprocess.run(
        [
            'gmx', 'pdb2gmx',
            '-f', pdb_file,
            '-o', output_name,
            '-ignh',
            '-ff', 'amber99sb',
            '-water', 'none',
            '-ss', 'no'
        ],
        input='1\n',  # Automatically select option 1
        text=True
    ) # Run the GROMACS pdb2gmx command with force field selection via stdin
    for backup_file in glob.glob('#*'): # Remove backup files like #filename#
        os.remove(backup_file)
#####################################
pdb_files = glob.glob(os.path.join(aceh_dir, "*.pdb"))
for pdb_file in pdb_files:
    traj = md.load_pdb(pdb_file)
    num_atoms = traj.n_atoms
    folder_name = os.path.join(input_dir, f"atoms_{num_atoms}")
    os.makedirs(folder_name, exist_ok=True)
    dst_path = os.path.join(folder_name, os.path.basename(pdb_file))
    shutil.copy(pdb_file, dst_path)
#####################################
pdb_files = [f for f in os.listdir(aceh_125_dir) if f.endswith(".pdb")]
pdb_files = natsorted(pdb_files)
merged_lines = [] # Merge content with MODEL/ENDMDL and remove any 'END' lines in between
for i, pdb_file in enumerate(pdb_files, start=1):
    with open(os.path.join(aceh_125_dir, pdb_file), "r") as f:
        pdb_data = f.read().strip().splitlines()
    pdb_data = [line for line in pdb_data if not line.strip().startswith("END")] # Remove any 'END' lines
    merged_lines.append(f"MODEL     {i}")
    merged_lines.extend(pdb_data)
    merged_lines.append("ENDMDL")
with open("merged_frames.pdb", "w") as f:
    f.write("\n".join(merged_lines) + "\n")

input_path = 'merged_frames.pdb'
output_path = 'cleaned_merged_frames.pdb'
with open(input_path, 'r') as f:
    lines = f.readlines()
seen_model = False
cleaned_lines = []
for line in lines:
    if line.strip().startswith('MODEL') and '1' in line:
        if seen_model:
            continue  # skip the second MODEL 1
        seen_model = True
    cleaned_lines.append(line)
with open(output_path, 'w') as f:
    f.writelines(cleaned_lines)
#####################################
file_path = 'cleaned_merged_frames.pdb'
with open(file_path, 'r') as file:
    pdb_lines = file.readlines()
frames = []
current_frame = []
for line in pdb_lines:
    current_frame.append(line)
    if line.startswith('ENDMDL'):
        frames.append(current_frame)
        current_frame = []
output_dir = aceh_125_reset_dir
for i, frame in enumerate(frames): # Save each frame into its own file
    frame_file_path = os.path.join(output_dir, f'frame_{i+1}.pdb')
    with open(frame_file_path, 'w') as frame_file:
        frame_file.write(''.join(frame))
#####################################
input_dir = aceh_125_reset_dir
output_dir = aceh_125_relax_dir
forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
with open(energy_file, 'w') as ef:
    ef.write("# FrameIndex Energy_kJ_per_mol\n")  # Header

    # Loop over PDB files
    for filename in os.listdir(input_dir):
        if not filename.endswith('.pdb'):
            continue

        match = re.search(r'frame_(\d+)', filename)
        if not match:
            print(f"Warning: Could not extract frame index from {filename}")
            continue
        frame_index = int(match.group(1))

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load structure
        pdb = PDBFile(input_path)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds
        )
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        simulation.minimizeEnergy()

        # Get minimized energy
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        # Save structure
        with open(output_path, 'w') as out_pdb:
            PDBFile.writeFile(simulation.topology, state.getPositions(), out_pdb)

        # Save energy to single file
        ef.write(f"{frame_index} {energy:.4f}\n")
##################################################
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
    ca_res_2_atoms = [7, 9, 16]

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
input_dir = aceh_125_relax_dir
output_dir = aceh_125_local_relax_dir
os.makedirs(output_dir, exist_ok=True)
# Process all PDB files
for frame_file in sorted(os.listdir(input_dir)):
    if not frame_file.endswith('.pdb'):
        continue
    input_file = os.path.join(input_dir, frame_file)
    output_file = os.path.join(output_dir, frame_file)

    reset_frame_to_local(input_file, output_file)
##################################################
input_dir = aceh_125_local_relax_dir
target_resid = 9

atom_targets = {
    'SG': f'{base_path}/s10_local_coordinates_str2str.npy',
    'CA': f'{base_path}/Ca10_local_coordinates_str2str.npy'
}

atom_coords = {atom_name: [] for atom_name in atom_targets}

def parse_pdb_frame(file_path):
    """Parse a .pdb file and return a list of (resid, atom_name, xyz) tuples."""
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
    atoms = parse_pdb_frame(pdb_file)

    found = {key: False for key in atom_targets}
    for resid, atom_name, coord in atoms:
        if resid == target_resid and atom_name in atom_targets:
            atom_coords[atom_name].append(coord)
            found[atom_name] = True

    for atom_name, was_found in found.items():
        if not was_found:
            print(f"Warning: {atom_name} of residue 10 missing in {frame_file}")

# Save each set of coordinates
for atom_name, coords in atom_coords.items():
    coords_array = np.array(coords)
    np.save(atom_targets[atom_name], coords_array)
##################################################
# Load Ca and S coordinates for Str2Str (reference)
ca_data = np.load(f'{base_path}/Ca10_local_coordinates_str2str.npy')
s_data = np.load(f'{base_path}/s10_local_coordinates_str2str.npy')

# Load Ca and S coordinates for Simulation (model)
ca_model = np.load(f'{MD_path}/Ca10_local_coordinates_simulation.npy')
s_model = np.load(f'{MD_path}/s10_local_coordinates_simulation.npy')

assert ca_data.shape == s_data.shape, "Mismatch between Ca and S data shapes!"
assert ca_model.shape == s_model.shape, "Mismatch between Ca and S model shapes!"
combined_data = np.hstack([ca_data, s_data])      # Shape (n_frames, 6)
combined_model = np.hstack([ca_model, s_model])   # Shape (n_frames, 6)
tree = cKDTree(combined_model)
distances, indices = tree.query(combined_data, k=10)
##################################################
top10_closest_frame_numbers = indices # Load the top 10 closest frame numbers for each frame
top10_closest_frame_numbers = top10_closest_frame_numbers + 1  # Convert to 1-based indexing if necessary
input_dir = f'{MD_path}/individual_frames_gro_MD_local_reset'
for frame_idx, candidate_frames in enumerate(top10_closest_frame_numbers, start=1):
    frame_folder = os.path.join(output_base_dir, f'frame_{frame_idx}') # Create a folder for each frame (e.g., frame_1, frame_2, etc.)
    os.makedirs(frame_folder, exist_ok=True)
    
    for rank, frame_number in enumerate(candidate_frames, start=1): # Copy the 10 closest candidate frames into this folder
        frame_filename = f'frame_{frame_number}.gro'
        src_file = os.path.join(input_dir, frame_filename)

        if os.path.exists(src_file):
            dest_file = os.path.join(frame_folder, f'candidate_{rank}_frame_{frame_number}.gro')
            shutil.copy(src_file, dest_file)
            print(f"Copied: {src_file} -> {dest_file}")
        else:
            print(f"Warning: {src_file} does not exist — skipping.")
##################################################
for frame_folder in natsorted(os.listdir(output_base_gro_dir)):# Loop through each frame_X subfolder
    frame_folder_path = os.path.join(output_base_gro_dir, frame_folder)
    if not os.path.isdir(frame_folder_path):
        continue  # Skip if not a folder
    output_folder_path = os.path.join(output_base_pdb_dir, frame_folder) # Create matching subfolder in the output directory
    os.makedirs(output_folder_path, exist_ok=True)

    for gro_file in natsorted(os.listdir(frame_folder_path)): # Process each .gro file inside this subfolder
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
##################################################
# Define the folders
aa_folder = aceh_125_local_relax_dir
sim_base_folder = output_base_pdb_dir
output_base_folder = assemble_MD

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

        candidate_index = sim_filename.split('_')[-1].split('.')[0]
        output_filepath = os.path.join(output_folder, f'candidate_{candidate_index}_AA.pdb')
        
        header_lines = update_residue_index_to_zero(header_lines)
        H_lines = update_residue_index_to_zero(H_lines)

        combined_lines = header_lines + H_lines + aa_lines  # Start with header + AA lines

        ###
        combined_lines[53] = replace_pdb_lines(combined_lines[53], line_37)
        combined_lines[141] = replace_pdb_lines(combined_lines[141], line_22)
        
        # remove extra Hs
        del combined_lines[172]
        del combined_lines[171]
        del combined_lines[170]
        del combined_lines[169]
        del combined_lines[168]
        del combined_lines[167]
        del combined_lines[166]
        del combined_lines[142]
        del combined_lines[54]
        
        combined_lines = renumber_atom_indices(combined_lines)
        combined_lines = fix_and_replace_ter_line(combined_lines)
        
        with open(output_filepath, 'w') as out_file:
            out_file.writelines(combined_lines)
##################################################
# Parameters
clash_distance_threshold = 0.9  # in nm
c_s_distance_threshold = 2.2    # in nm
s_index = 137  # 0-based
c_index = 134  # cb

# Paths
input_base_folder = assemble_MD
output_base_folder = screen_assemble_MD
kept_list_file = f'{base_path}/kept_structures.txt'

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

########################################
kept_frames_file = f'{base_path}/kept_frames.txt'
frame_labels = set()

# Read and collect unique frame labels
with open(kept_list_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("frame_"):
            frame = line.split('/')[0]  # e.g., 'frame_4'
            frame_labels.add(frame)

# Sort and write to output
with open(kept_frames_file, 'w') as out:
    for frame in sorted(frame_labels):
        out.write(frame + '\n')

# Read frame labels
with open(kept_frames_file, 'r') as f:
    frames = [line.strip() for line in f if line.strip()]

# Natural sort
sorted_frames = natsorted(frames)

# Overwrite the file with sorted frames
with open(kept_frames_file, 'w') as f:
    for frame in sorted_frames:
        f.write(frame + '\n')
########################################
# Paths
source_root = screen_assemble_MD
# Read kept frame names
with open(kept_frames_file, 'r') as f:
    kept_frames = [line.strip() for line in f if line.strip()]
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
#####################################
# Paths
selected_dir = destination_root
source_gro_dir = f"{MD_path}/individual_frames_gro_MD_local_reset"
target_gro_dir = MD_lib
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