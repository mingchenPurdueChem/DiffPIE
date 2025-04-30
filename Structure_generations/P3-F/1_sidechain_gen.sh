#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p data/individual_frames_pdb_str2str_local_reset_AA

# Loop through all PDB files and run FASPR
for pdb_file in data/individual_frames_pdb_str2str_local_reset/*.pdb; do
    base_name=$(basename "$pdb_file")
    output_name="data/individual_frames_pdb_str2str_local_reset_AA/${base_name%.pdb}_AA.pdb"
    ./FASPR -i "$pdb_file" -o "$output_name"
done
