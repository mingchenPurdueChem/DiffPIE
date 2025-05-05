import os
import shutil
import numpy as np
import shutil
import re
import glob
import subprocess
import mdtraj as md
from natsort import natsorted
import math
import torch
import torch.nn.functional as F
##################################
##################################
base_path = './Data'
os.makedirs(base_path, exist_ok=True)
cache_dir = f"{base_path}/Cache"
os.makedirs(cache_dir, exist_ok=True)
################################## # Load the PDB file to remove GLY and to set to individual frames
str2str_path = "Str2Str/logs/inference/runs/Au_F_2/samples/all_delta/"
str2str_pdb_list = glob.glob(os.path.join(str2str_path, '*.pdb'))
str2str_pdb = str2str_pdb_list[0]
file_path = os.path.join(base_path, 'Str2Str.pdb')
shutil.copy(str2str_pdb, file_path)
with open(str2str_pdb, "r") as infile, open(file_path, "w") as outfile:
    for line in infile:
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "GLY":
            continue
        outfile.write(line)
individual_dir = os.path.join(base_path, 'individual')
os.makedirs(individual_dir, exist_ok=True)
################################## generate side chain
aa_dir = os.path.join(base_path, 'individual_AA')
os.makedirs(aa_dir, exist_ok=True)
################################## generate H
gro_dir = f"{base_path}/GROs"
os.makedirs(gro_dir, exist_ok=True)
################################## first attempt at d=0.5 nm
intital_dir = os.path.join(base_path, 'first_merge')
os.makedirs(intital_dir, exist_ok=True)
##################################
gol_file_path =  "Gol.gro"
if not os.path.isdir(pot_dir):
    raise FileNotFoundError(f"Directory '{Gol}' does not exist.")
pot_dir = './One_Ca'
if not os.path.isdir(pot_dir):
    raise FileNotFoundError(f"Directory '{pot_dir}' does not exist.")
MD_traj = './Traj'
if not os.path.isdir(MD_traj):
    raise FileNotFoundError(f"Directory '{MD_traj}' does not exist.")
##################################
results_dir = "./Results" 
results_dir_merged = "./Results/merged" 
os.makedirs(results_dir_merged, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
##################################
##################################
sequence = 'KLVFFAE'
sequence_list = list(sequence)
res_names = ['GLU','LEU','LYS','PHE','VAL']
# Define indices for each residue manually (adjust as needed)
# here SC include all atoms in the residual 
# default values as from MD of dipepetide
# the indecis already from 0
MD_res_indices = {
    'GLU': {
        'Ca_indices': np.array([0, 8, 23]),  # Example indices for GLU
        'SC_indices': np.arange(6, 21)      # Example indices for GLU side chain
    },
    'LEU': {
        'Ca_indices': np.array([0, 8, 27]), # Example indices for LEU
        'SC_indices': np.arange(6, 25)      # Example indices for LEU side chain
    },
    'LYS': {
        'Ca_indices': np.array([0, 8, 30]), # Example indices for LYS
        'SC_indices': np.arange(6, 28)      # Example indices for LYS side chain
    },
    'PHE': {
        'Ca_indices': np.array([0, 8, 28]), # Example indices for PHE
        'SC_indices': np.arange(6, 26)      # Example indices for PHE side chain
    },
    'VAL': {
        'Ca_indices': np.array([0, 8, 24]), # Example indices for VAL
        'SC_indices': np.arange(6, 22)      # Example indices for VAL side chain
    }
}
######################################## Str2Str
# sequence LYS-LEU-VAL-PHE-PHE-ALA-GLU
# Define original Ca indices, start from 1, PDB
# -1 means ends
# here SC include all atoms in the residual 
peptide_sc = [1,2,3,4,5,7]
Ca_indices_original = {
    1: {'res_name': 'LYS', 'Ca_indices': np.array([-1, 3139, 3161]), 'SC_indices': np.arange(3137, 3159)},
    2: {'res_name': 'LEU', 'Ca_indices': np.array([3139, 3161, 3180]), 'SC_indices': np.arange(3159, 3178)},
    3: {'res_name': 'VAL', 'Ca_indices': np.array([3161, 3180, 3196]), 'SC_indices': np.arange(3178, 3194)},
    4: {'res_name': 'PHE', 'Ca_indices': np.array([3180, 3196, 3216]), 'SC_indices': np.arange(3194, 3214)},
    5: {'res_name': 'PHE', 'Ca_indices': np.array([3196, 3216, 3236]), 'SC_indices': np.arange(3214, 3234)},
    7: {'res_name': 'GLU', 'Ca_indices': np.array([3236, 3246, -1]), 'SC_indices': np.arange(3244, 3259)},
}
S2S_residue_indices = {
    res_num: {'res_name': info['res_name'], 'Ca_indices': info['Ca_indices'] - 1, 'SC_indices': info['SC_indices'] - 1}
    for res_num, info in Ca_indices_original.items()
}
##################################
##################################
with open(file_path, 'r') as file:
    pdb_lines = file.readlines()
frames = []
current_frame = []
for line in pdb_lines:
    current_frame.append(line)
    if line.startswith('ENDMDL'): # Split frames using ENDMDL as separator
        frames.append(current_frame)
        current_frame = []
for i, frame in enumerate(frames):
    frame_file_path = os.path.join(output_dir, f'frame_{i+1}.pdb')# Save each frame into its own file
    with open(frame_file_path, 'w') as frame_file:
        frame_file.write(''.join(frame))
##################################       
pdb_files = glob.glob(os.path.join(output_dir, '*.pdb')) # Loop through all PDB files in individual files
for pdb_file in pdb_files:
    base_name = os.path.basename(pdb_file)
    output_name = os.path.join(aa_dir, base_name.replace('.pdb', '_AA.pdb'))
    subprocess.run(['./FASPR', '-i', pdb_file, '-o', output_name])
##################################
##################################
pdb_files = [f for f in os.listdir(aa_dir) if f.endswith(".pdb")]
for pdb_file in pdb_files:
    pdb_path = os.path.join(aa_dir, pdb_file)
    gro_file = pdb_file.replace(".pdb", ".gro")
    gro_path = os.path.join(gro_dir, gro_file)

    # Run GROMACS editconf to convert PDB to GRO
    command = f"gmx editconf -f {pdb_path} -o {gro_path}"
    subprocess.run(command, shell=True, check=True)
########################################### add H
for file in glob.glob(os.path.join(gro_dir, "#*#")):
    os.remove(file)

for gro_file in os.listdir(gro_dir):
    for file in glob.glob(os.path.join(os.getcwd(), "#*#")):
        os.remove(file)
    if gro_file.startswith("frame") and gro_file.endswith(".gro"):
        input_path = os.path.join(gro_dir, gro_file)
        output_path = os.path.join(gro_dir, f"{gro_file[:-4]}_processed.gro")
        cmd = [
            "gmx", "pdb2gmx", 
            "-f", input_path, 
            "-o", output_path, 
            "-ff", "oplsaa", 
            "-water", "spc"
        ]
        subprocess.run(cmd)
########################################### move to interface with com at 4 nm
Au_PBC = [4.10200, 4.05993, 4.95517]  # Gold periodic boundary conditions update if you changed the system size
Au_Z = 4.0  # Gold surface at 4 nm
dis2Au = 0.5  # Buffer zone due to soft-core repulsion
target_Z = Au_Z - dis2Au  # Target COM height

gro_files = [f for f in os.listdir(gro_dir) if f.endswith("_processed.gro")]
with open(gol_file_path, "r") as f:
    gol_lines = f.readlines()
header = gol_lines[0]  # Title line
box_size = gol_lines[-1]  # Box dimensions
num_atoms_gol = int(gol_lines[1].strip())  # Extract atom count from Gol.gro
gol_molecules = gol_lines[2:-1]  # Exclude header and box size line

for gro_file in natsorted(gro_files):
    raw_file_path = os.path.join(gro_dir, gro_file)
    traj = md.load(raw_file_path)
    ###
    masses = np.array([atom.element.mass for atom in traj.topology.atoms]) # Compute center of mass (COM)
    com = np.sum(traj.xyz[0] * masses[:, np.newaxis], axis=0) / np.sum(masses)
    com_Z = com[2]
    index = int(gro_file.split('_')[1])
    trans_Z = target_Z - com_Z
    ###
    peptide_xyz = traj.xyz     # Apply translation
    peptide_xyz[:, :, 2] += trans_Z
    peptide_xyz[:, :, 0] += Au_PBC[0] / 2
    peptide_xyz[:, :, 1] += Au_PBC[1] / 2
    ###
    transformed_gro_path = os.path.join(cache_dir, f"{gro_file.replace('_processed.gro', '_trans.gro')}") # Save translated structure
    traj.save_gro(transformed_gro_path)
    with open(transformed_gro_path, "r") as f:  # Merge with Gol.gro
        klv_lines = f.readlines()
    klv_molecules = klv_lines[2:-1]  # Exclude header and box size line
    total_atoms = num_atoms_gol + len(klv_molecules)  # Update atom count
    merged_file_path = os.path.join(gro_dir, gro_file.replace("_processed.gro", "_Gol_merged.gro"))
    ###
    with open(merged_file_path, "w") as f:
        f.write(header)  # Keep original title
        f.write(f" {total_atoms}\n")  # Update total atom count
        f.writelines(gol_molecules)  # Write Gol molecules
        f.writelines(klv_molecules)  # Append KLV molecules
        f.write(box_size)  # Write box dimensions
########################################### calculate COM
for gro_file in glob.glob(os.path.join(gro_dir, '*_Gol_merged.gro')):
    shutil.move(gro_file, intital_dir)
target_resids = list(range(2, 9))
ca_atom_name = 'CA'

all_frames_ca_coords = []
def parse_gro_frame(file_path):
    """Parse a .gro file and return a list of (resid, atom_name, xyz) tuples."""
    atoms = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[2:-1]:  # skip box line
            resid = int(line[0:5])
            atom_name = line[10:15].strip()
            x = float(line[20:28])  # convert from nm to Å
            y = float(line[28:36])
            z = float(line[36:44])
            atoms.append((resid, atom_name, np.array([x, y, z])))
    return atoms
for frame_file in natsorted(os.listdir(intital_dir)):
    if not frame_file.endswith('.gro'):
        continue
    gro_file = os.path.join(intital_dir, frame_file)
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
distance2Au = 4-ca_all_array # Au at 4 nm
np.save('Ca2Au_residues_1_to_7.npy', distance2Au)
########################################### adjust Z
def interpolate_logP(coords, logP_tensor):
    """
    Interpolates logP using bilinear or trilinear interpolation with PyTorch.
    Handles both 2D and 3D cases dynamically.
    """
    dim = 1
    ranges = [(0.1, 3.5)] * dim
    coords = torch.as_tensor(coords, dtype=torch.float32, device=logP_tensor.device # Ensure coords is torch tensor on same device as logP_tensor
    # Normalize to [-1, 1]
    mins = torch.tensor([r[0] for r in ranges], dtype=coords.dtype, device=coords.device)
    spans = torch.tensor([r[1] - r[0] for r in ranges], dtype=coords.dtype, device=coords.device)
    coords_normalized = (coords - mins) / spans
    coords_normalized = 2.0 * coords_normalized - 1.0
    coords_normalized = torch.clamp(coords_normalized, -1.0, 1.0)
    logP_tensor = logP_tensor.view(1, 1, 1, -1)
    coords_normalized = coords_normalized.view(1, 1, -1, 1)
    grid = torch.cat([coords_normalized, torch.zeros_like(coords_normalized)], dim=-1)  # [1, 1, N, 2]
    sampled = F.grid_sample(logP_tensor, grid, mode='bilinear', align_corners=True)
    return sampled.view(-1)
def main(data,res):
    sequence_to_file = {}
    file_name = f"{sequence_list[res]}_logP1D.pt"
    sequence_to_file[sequence_list[res]] = os.path.join(pot_dir, file_name)  # Create a mapping dictionary for the sequence
    pot_name = sequence_to_file[sequence_list[res]] # Load saved logP data from .pt file
    logP_tensor = torch.load(pot_name).float()
    data_torch = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    predicted_logP = interpolate_logP(data_torch,logP_tensor)
    return predicted_logP

kBT_kcal = 0.593 # Thermal energy unit does not matter
adjust_z_range = np.linspace(-0.6, 0.6, 21)  # from -1.0 to +1.0 nm in 0.05 nm steps
distance2Au = np.load('Ca2Au_residues_1_to_7.npy')
best_energies = []
best_adjusts = []
for frame_idx in range(distance2Au.shape[0]):
    min_energy = np.inf
    best_shift = None
    for z_shift in adjust_z_range:
        frame_logp = []
        for res_idx in range(7):
            coord = distance2Au[frame_idx, res_idx].copy()
            coord[2] += z_shift  # apply z-shift to all residues
            try:
                logp_value = main(coord[2], res_idx)
                frame_logp.append(logp_value.detach().numpy())
            except Exception as e:
                frame_logp.append(np.nan)
        frame_logp = np.array(frame_logp)
        total_logp = np.nansum(frame_logp)
        total_energy = -kBT_kcal * total_logp
        if total_energy < min_energy:
            min_energy = total_energy
            best_shift = z_shift
    best_energies.append(min_energy)
    best_adjusts.append(best_shift)
best_energies = np.array(best_energies)  # shape: (n_frames,)
best_adjusts = np.array(best_adjusts)    # shape: (n_frames,)
########################################### move to interface with adjustment
Au_PBC = [4.10200, 4.05993, 4.95517]  # Gold periodic boundary conditions
Au_Z = 4.0  # Gold surface at 4 nm
dis2Au = 0.5  # Buffer zone due to soft-core repulsion
target_Z = Au_Z - dis2Au  # Target COM height

gro_files = [f for f in os.listdir(gro_dir) if f.endswith("_processed.gro")]
with open(gol_file_path, "r") as f:
    gol_lines = f.readlines()
header = gol_lines[0]  # Title line
box_size = gol_lines[-1]  # Box dimensions
num_atoms_gol = int(gol_lines[1].strip())  # Extract atom count from Gol.gro
gol_molecules = gol_lines[2:-1]  # Exclude header and box size line

for gro_file in natsorted(gro_files):
    raw_file_path = os.path.join(gro_dir, gro_file)
    traj = md.load(raw_file_path)
    ###
    masses = np.array([atom.element.mass for atom in traj.topology.atoms]) # Compute center of mass (COM)
    com = np.sum(traj.xyz[0] * masses[:, np.newaxis], axis=0) / np.sum(masses)
    com_Z = com[2]
    index = int(gro_file.split('_')[1])-1
    z_adjust_frame = best_adjusts[index]
    trans_Z = target_Z - com_Z - z_adjust_frame
    ###
    peptide_xyz = traj.xyz     # Apply translation
    peptide_xyz[:, :, 2] += trans_Z
    peptide_xyz[:, :, 0] += Au_PBC[0] / 2
    peptide_xyz[:, :, 1] += Au_PBC[1] / 2
    ###
    transformed_gro_path = os.path.join(cache_dir, f"{gro_file.replace('_processed.gro', '_trans.gro')}") # Save translated structure
    traj.save_gro(transformed_gro_path)
    with open(transformed_gro_path, "r") as f:  # Merge with Gol.gro
        klv_lines = f.readlines()
    klv_molecules = klv_lines[2:-1]  # Exclude header and box size line
    total_atoms = num_atoms_gol + len(klv_molecules)  # Update atom count
    merged_file_path = os.path.join(gro_dir, gro_file.replace("_processed.gro", "_Gol_merged.gro"))
    ###
    with open(merged_file_path, "w") as f:
        f.write(header)  # Keep original title
        f.write(f" {total_atoms}\n")  # Update total atom count
        f.writelines(gol_molecules)  # Write Gol molecules
        f.writelines(klv_molecules)  # Append KLV molecules
        f.write(box_size)  # Write box dimensions
########################################### remove teriminal Hs
gro_files = [f for f in os.listdir(gro_dir) if f.endswith("_Gol_merged.gro")]
indices_to_remove = np.array([3261, 3138, 3139]) - 1  # Convert to 0-based indexing

def run_gmx_editconf(input_gro, output_gro):
    command = ['gmx', 'editconf', '-f', input_gro, '-o', output_gro]
    subprocess.run(command, check=True)

for gro_file in gro_files:
    gro_file_path = os.path.join(gro_dir, gro_file)
    traj = md.load(gro_file_path)
    ###
    indices_to_remove_valid = indices_to_remove[indices_to_remove < traj.n_atoms]
    mask = np.ones(traj.n_atoms, dtype=bool)
    mask[indices_to_remove_valid] = False
    ###
    xyz = traj.xyz[:, mask, :]
    topology = traj.topology.subset(np.where(mask)[0])
    traj_truncated = md.Trajectory(xyz, topology)
    ###
    truncated_gro_path = os.path.join(cache_dir, gro_file.replace(".gro", "_truncated.gro"))
    traj_truncated.save_gro(truncated_gro_path)
    ###
    final_gro_path = os.path.join(cache_dir, gro_file.replace(".gro", "_reset.gro"))
    run_gmx_editconf(truncated_gro_path, final_gro_path)
################################## side chain generation
##################################    
# get the xyz intact early is neat but slow
def load_MD(res_name):
    gro_file = f"./Traj/package_{res_name}/protein.gro"
    xtc_file = f"./Traj/package_{res_name}/traj_comp.xtc"
    # Load trajectory
    traj = md.load(xtc_file, top=gro_file)
    PBC = traj.unitcell_lengths  # Shape: (n_frames, 3) → (Lx, Ly, Lz) per frame
    # Extract relevant atoms
    Ca_indices = MD_res_indices[res_name]['Ca_indices']
    SC_indices = MD_res_indices[res_name]['SC_indices']
    xyz_Ca = traj.atom_slice(Ca_indices).xyz  # Shape: (n_frames, len(Ca_indices), 3)
    xyz_SC = traj.atom_slice(SC_indices).xyz  # Shape: (n_frames, len(SC_indices), 3)
    ###### **Screening Ca based on z-coordinate** ######
    distance_xyz_Ca = xyz_Ca[:, :, 2] - 0.1
    mask = np.all(distance_xyz_Ca < 2.5, axis=1)  # Retain frames where all Ca atoms have z < 1.5 here need adjust according to S2S!!!!!!!
    filtered_xyz_Ca = xyz_Ca[mask]
    filtered_xyz_SC = xyz_SC[mask]
    filtered_PBC = PBC[mask]  # Keep only selected frames' PBC in case of NPT with changing box
    ###### **PBC Correction** ######
    ref_atom_x = filtered_xyz_SC[:, 0, 0]  # Reference x per frame
    ref_atom_y = filtered_xyz_SC[:, 0, 1]  # Reference y per frame
    # Apply periodic boundary corrections per frame
    for atom in range(filtered_xyz_SC.shape[1]):
        # Correct x-coordinates
        mask_x_low = filtered_xyz_SC[:, atom, 0] + filtered_PBC[:, 0] / 2 < ref_atom_x
        mask_x_high = filtered_xyz_SC[:, atom, 0] - filtered_PBC[:, 0] / 2 > ref_atom_x
        filtered_xyz_SC[mask_x_low, atom, 0] += filtered_PBC[mask_x_low, 0]
        filtered_xyz_SC[mask_x_high, atom, 0] -= filtered_PBC[mask_x_high, 0]
        # Correct y-coordinates
        mask_y_low = filtered_xyz_SC[:, atom, 1] + filtered_PBC[:, 1] / 2 < ref_atom_y
        mask_y_high = filtered_xyz_SC[:, atom, 1] - filtered_PBC[:, 1] / 2 > ref_atom_y
        filtered_xyz_SC[mask_y_low, atom, 1] += filtered_PBC[mask_y_low, 1]
        filtered_xyz_SC[mask_y_high, atom, 1] -= filtered_PBC[mask_y_high, 1]
    ###### **Compute CA Distance** ######
    CA_dis = filtered_xyz_Ca[:, :, 2] - 0.1
    return CA_dis, filtered_xyz_SC   
################################## 
class LocalGlobal:
    def __init__(self, s2s_SC_indices):
        N_Ca_C_idx = [s2s_SC_indices[0],s2s_SC_indices[2],s2s_SC_indices[-2]]
        str2str_N_Ca_C = traj_flipped.xyz[:, N_Ca_C_idx, :]
        origin, axes = self.build_local_frame_all(str2str_N_Ca_C[:,0], str2str_N_Ca_C[:,1], str2str_N_Ca_C[:,2])
        self.str2str_local = self.transform_to_local(str2str_N_Ca_C)
        self.origin = origin
        self.axes = axes
        
    def transform_to_local(self, coords, batch_size=100000):
        num_residues = coords.shape[0]
        transformed_coords = np.zeros_like(coords)
        for i in range(0, num_residues, batch_size):
            batch_coords = coords[i:i+batch_size]  # Shape: (batch_size, 3, 3)
            n = batch_coords[:, 0]  # Shape: (batch_size, 3)
            ca = batch_coords[:, 1]  # Shape: (batch_size, 3)
            c = batch_coords[:, 2]  # Shape: (batch_size, 3)
            # Compute local frame for each residue in batch
            origin, axes = self.build_local_frame_all(n, ca, c)  # origin: (batch_size, 3), axes: (batch_size, 3, 3)
            # Reshape origin for broadcasting
            origin = origin[:, np.newaxis, :]  # Shape: (batch_size, 1, 3)
            # Apply transformation using batch-wise matrix multiplication
            transformed_coords[i:i+batch_size] = np.einsum('bij,bjk->bik', batch_coords - origin, axes.transpose(0, 2, 1))
        return transformed_coords

    def build_local_frame_all(self, n, ca, c):
        x_axis = n - ca  # Shape: (batch_size, 3)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)  # Normalize
        z_axis = np.cross(x_axis, c - ca)  # Shape: (batch_size, 3)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)  # Normalize
        y_axis = np.cross(z_axis, x_axis)  # Shape: (batch_size, 3)
        axes = np.stack([x_axis, y_axis, z_axis], axis=1)  # Shape: (batch_size, 3, 3)
        return ca, axes

    def transform_to_local_MD(self, coords, batch_size=100000):
        num_residues = coords.shape[0]
        transformed_coords = np.zeros_like(coords)
        for i in range(0, num_residues, batch_size):
            batch_coords = coords[i:i+batch_size]  # Shape: (batch_size, 3, 3)
            n = batch_coords[:, 0]  # Shape: (batch_size, 3)
            ca = batch_coords[:, 2]  # Shape: (batch_size, 3)
            c = batch_coords[:, -2]  # Shape: (batch_size, 3)
            # Compute local frame for each residue in batch
            origin, axes = self.build_local_frame_all(n, ca, c)  # origin: (batch_size, 3), axes: (batch_size, 3, 3)
            # Reshape origin for broadcasting
            origin = origin[:, np.newaxis, :]  # Shape: (batch_size, 1, 3)
            # Apply transformation using batch-wise matrix multiplication
            transformed_coords[i:i+batch_size] = np.einsum('bij,bjk->bik', batch_coords - origin, axes.transpose(0, 2, 1))
        return transformed_coords  
        
    def transform_back_to_global(self, local_coords, origin, axes):
        if axes.ndim == 2:  # If shape is (3,3), expand to (1,3,3)
            axes = np.expand_dims(axes, axis=0)
        if origin.ndim == 2:  # If shape is (batch_size,3), expand to (batch_size,1,3)
            origin = origin[:, np.newaxis, :]
        # Inverse transformation: Multiply with `axes` (not transposed) and add back origin
        global_coords = np.einsum('bij,bjk->bik', local_coords, axes) + origin
        return global_coords
################################## 
class SidechainReplacement:
        def __init__(self, idx):
            self.Ca_indices = S2S_residue_indices[idx]['Ca_indices']
            self.SC_indices = S2S_residue_indices[idx]['SC_indices']
            self.res_name = S2S_residue_indices[idx]['res_name']
            self.xyz = traj_flipped.xyz[:, self.Ca_indices, :].squeeze(0)
            self.distance = self.xyz[:, 2] - Au_shift
            self.idx = idx
            self.LG = LocalGlobal(self.SC_indices)
            self.origin = self.LG.origin
            self.axes = self.LG.axes

        def local_info(self, MD_SC_xyz):
            str2str_C_local = self.LG.str2str_local[0][:, 2]
            MD_SC_local = self.LG.transform_to_local_MD(MD_SC_xyz)
            MD_C_local = MD_SC_local[:, -2, :]
            c_align = np.linalg.norm(MD_C_local - str2str_C_local, axis=1)
            MD_SC_global = self.LG.transform_back_to_global(MD_SC_local, self.origin, self.axes)
            return c_align, MD_SC_global

        def sidechain(self, best_sc):
            SC_indices = S2S_residue_indices[self.idx]['SC_indices']
            filtered_tra_SC = traj_flipped.atom_slice(SC_indices)
            old_sc = filtered_tra_SC.xyz
            new_sc = np.expand_dims(best_sc, axis=0)  # Shape (1, n_atoms, 3)
            align_sc = old_sc
            align_sc[:, 4:-2, :] = new_sc[:, 4:-2, :]
            xyz_flipped[:, SC_indices, :] = align_sc
################################## 
gro_files = [f for f in os.listdir(cache_dir) if f.endswith("_Gol_merged_reset.gro")]
gro_files = natsorted(gro_files)

Au_shift = 0.1 # Gold layer shift 
Au_Z = 4.0 # Gold surface at 4.0 nm
z_min = 0.10 
z_max_ad = 0.80 
align_weights = [1, 2]

for gro_file in natsorted(gro_files):
    index = int(gro_file.split('_')[1])
    gro_file_path = os.path.join(cache_dir, gro_file)
    base_name = gro_file.replace("_Gol_merged_reset.gro", "")
    traj_new = md.load(gro_file_path) # Load trajectory
    xyz_flipped = traj_new.xyz.copy() # Flip coordinates along z-axis
    xyz_flipped[:, :, 2] *= -1  # Flip z-coordinates
    xyz_flipped[:, :, 2] += (Au_Z + Au_shift)  # Align with gold surface
    traj_flipped = traj_new
    traj_flipped.xyz = xyz_flipped
    flipped_gro_path = os.path.join(cache_dir, gro_file.replace("_Gol_merged_reset.gro", "_flipped.gro")) # Save flipped structure
    traj_flipped.save_gro(flipped_gro_path)

    for idx in peptide_sc:
        sr = SidechainReplacement(idx)
        S2S_CA_dis = sr.distance.reshape(1, 3)
        res_name = sr.res_name
        MD_CA_dis, MD_SC_xyz = load_MD(res_name) # Load MD data for sidechain alignment
        if idx == peptide_sc[0]: # Align based on Ca-Ca-Ca distances
            norm_align = np.linalg.norm(MD_CA_dis[:, 1:] - S2S_CA_dis[:, 1:], axis=1)
        elif idx == peptide_sc[-1]:
            norm_align = np.linalg.norm(MD_CA_dis[:, :-1] - S2S_CA_dis[:, :-1], axis=1)
        else:
            norm_align = np.linalg.norm(MD_CA_dis - S2S_CA_dis, axis=1)
        c_align, MD_SC_global = sr.local_info(MD_SC_xyz) # Align using N_Ca_C local info
        align = norm_align * align_weights[0] + c_align * align_weights[1]
        z_max = S2S_CA_dis[:, 1] + z_max_ad # Z-cutoff filtering
        MD_SC_global_z = MD_SC_global[:, 4:-2, 2]
        valid_indices = np.where(
            (np.all(MD_SC_global_z > z_min, axis=1)) & (np.all(MD_SC_global_z < z_max, axis=1))
        )[0]
        if valid_indices.size > 0:
            index_best = valid_indices[np.argmin(align[valid_indices])]
            #print(f"Min Z in valid frames: {np.min(MD_SC_global_z[valid_indices])}")
        else:
            index_best = align.argmin()  # Fallback: Choose best overall
            print('cannot find a match')
        best_sc = MD_SC_global[index_best, :, :]
        print(f"Best sidechain index: {index_best}")
        sr.sidechain(best_sc)

    final_gro_path = os.path.join(results_dir, gro_file.replace("_Gol_merged_reset.gro", "_reconstructed.gro")) # Save final reconstructed file
    traj_flipped.xyz = xyz_flipped
    traj_flipped.save_gro(final_gro_path)

    # Split into Au and Peptide structures
    Au = traj_flipped.atom_slice(np.arange(0, 3136))
    box_size = np.array([[4.10200, 4.05993, 4.95517]])
    box_angles = np.array([[90.0, 90.0, 90.0]])
    Au.unitcell_angles = box_angles
    Au.unitcell_lengths = box_size
    Pep = traj_flipped.atom_slice(np.arange(3136, 3258))
    au_gro_path = os.path.join(results_dir, "Au.gro")
    pep_gro_path = os.path.join(results_dir, "Pep.gro")
    Au.save_gro(au_gro_path)
    Pep.save_gro(pep_gro_path)
    
    # Step 1: Generate index file to remove hydrogens
    index_file_path = os.path.join(results_dir, "index.ndx")
    subprocess.run(
        ["gmx", "make_ndx", "-f", pep_gro_path, "-o", index_file_path],
        input="q\n", text=True, check=True
    )
    # Step 2: Remove all hydrogen atoms (Select **group 2**)
    noH_gro_path = os.path.join(results_dir, "noH.gro")
    subprocess.run(
        ["gmx", "trjconv", "-s", pep_gro_path, "-f", pep_gro_path, "-o", noH_gro_path, "-novel", "-n", index_file_path],
        input="2\n", text=True, check=True
    )
    # Step 3: Re-add hydrogens using pdb2gmx with SPC water model
    h_gro_path = os.path.join(results_dir, "h.gro")
    subprocess.run(
        ["gmx", "pdb2gmx", "-f", noH_gro_path, "-o", h_gro_path, "-ff", "oplsaa", "-water", "spc"],
        check=True
    )
    for file in glob.glob(os.path.join(results_dir, "#*#")):
        os.remove(file)
    for file in glob.glob(os.path.join(os.getcwd(), "#*#")):
        os.remove(file)
    # Step 4: Merge Au and modified peptide into final system
    merged_gro_path = os.path.join(results_dir_merged, f"{base_name}_Gol_merged.gro")
    with open(au_gro_path, "r") as f:
        gol_lines = f.readlines()
    with open(h_gro_path, "r") as f:
        klv_lines = f.readlines()

    header = gol_lines[0]    # Extract headers, atom count, and box size
    box_size_line = gol_lines[-1]
    num_atoms_gol = int(gol_lines[1].strip())

    gol_molecules = gol_lines[2:-1]    # Extract molecular data
    klv_molecules = klv_lines[2:-1]
    total_atoms = num_atoms_gol + len(klv_molecules)

    with open(merged_gro_path, "w") as f: # Write final merged structure
        f.write(header)
        f.write(f" {total_atoms}\n")
        f.writelines(gol_molecules)
        f.writelines(klv_molecules)
        f.write(box_size_line)
################################## clear up
for ext in ('*.itp', '*.top'):
    for file in glob.glob(os.path.join(os.getcwd(), ext)):
        os.remove(file)
