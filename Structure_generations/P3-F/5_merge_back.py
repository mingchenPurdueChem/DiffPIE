import os
from natsort import natsorted

# Folder containing the PDB files
#folder = "individual_frames_pdb_str2str_local_reset_AA_with_H_relaxed"
folder = "/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset_ACE_with_H/atoms_125"

# Get all .pdb files and sort naturally
pdb_files = [f for f in os.listdir(folder) if f.endswith(".pdb")]
pdb_files = natsorted(pdb_files)

# Merge content with MODEL/ENDMDL and remove any 'END' lines in between
merged_lines = []
for i, pdb_file in enumerate(pdb_files, start=1):
    with open(os.path.join(folder, pdb_file), "r") as f:
        pdb_data = f.read().strip().splitlines()

    # Remove any 'END' lines
    pdb_data = [line for line in pdb_data if not line.strip().startswith("END")]

    merged_lines.append(f"MODEL     {i}")
    merged_lines.extend(pdb_data)
    merged_lines.append("ENDMDL")

# Save to a merged file
with open("merged_frames.pdb", "w") as f:
    f.write("\n".join(merged_lines) + "\n")

print(f"Merged {len(pdb_files)} PDB files into 'merged_frames.pdb' with MODEL/ENDMDL blocks and no 'END' in between.")


