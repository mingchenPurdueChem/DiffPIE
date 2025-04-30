import os
import re
from openmm.app import *
from openmm import *
from openmm.unit import *

input_dir = 'data/reset_index'
output_dir = 'data/individual_frames_pdb_str2str_local_reset_ACE_with_H_relaxed'
energy_file = 'data/peptide_energy.txt'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load force field
forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

# Open energy file for writing
with open(energy_file, 'w') as ef:
    ef.write("# FrameIndex Energy_kJ_per_mol\n")  # Header

    # Loop over PDB files
    for filename in os.listdir(input_dir):
        if not filename.endswith('.pdb'):
            continue

        # Extract frame index (e.g., "frame_2_AA_reset_H.pdb" -> 2)
        match = re.search(r'frame_(\d+)', filename)
        if not match:
            print(f"Warning: Could not extract frame index from {filename}")
            continue
        frame_index = int(match.group(1))

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f'Processing {filename}...')

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

        #print(f"Saved relaxed PDB and energy for frame {frame_index}")

print(f"\nAll energies written to: {energy_file}")
##################################################
import os
import numpy as np

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
input_dir = 'data/individual_frames_pdb_str2str_local_reset_ACE_with_H_relaxed'
output_dir = 'data/individual_frames_pdb_str2str_local_reset_AA_with_H_relaxed'

os.makedirs(output_dir, exist_ok=True)

# Process all PDB files
for frame_file in sorted(os.listdir(input_dir)):
    if not frame_file.endswith('.pdb'):
        continue
    input_file = os.path.join(input_dir, frame_file)
    output_file = os.path.join(output_dir, frame_file)

    reset_frame_to_local(input_file, output_file)

print(f"All frames processed and saved to: {output_dir}")
