import os
import mdtraj as md
import shutil
import glob

input_dir = "/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset_ACE_with_H"  # ✅ Set this to your actual folder
pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))

for pdb_file in pdb_files:
    try:
        traj = md.load_pdb(pdb_file)
        num_atoms = traj.n_atoms
        folder_name = os.path.join(input_dir, f"atoms_{num_atoms}")
        
        # Create output folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)
        
        # Destination path
        dst_path = os.path.join(folder_name, os.path.basename(pdb_file))
        
        # Copy the file
        shutil.copy(pdb_file, dst_path)
        
        print(f"✅ Moved {pdb_file} → {folder_name}/")
        
    except Exception as e:
        print(f"⚠️ Error loading {pdb_file}: {e}")