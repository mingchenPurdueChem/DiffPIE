#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p /home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset_AA

# Loop through all PDB files and run FASPR
for pdb_file in /home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset/*.pdb; do
    base_name=$(basename "$pdb_file")
    output_name="/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/individual_frames_pdb_str2str_local_reset_AA/${base_name%.pdb}_AA.pdb"
    ./FASPR -i "$pdb_file" -o "$output_name"
done
