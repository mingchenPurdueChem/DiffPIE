import os

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
            #print(f"Processed: {filename}")

# Example usage:
input_dir = "data/individual_frames_pdb_str2str_local_reset_AA"
output_dir = "data/individual_frames_pdb_str2str_local_reset_ACE"
process_all_pdbs(input_dir, output_dir)
