#!/bin/bash

input_dir="data/individual_frames_pdb_str2str_local_reset_ACE"
output_dir="data/individual_frames_pdb_str2str_local_reset_ACE_with_H"

mkdir -p "$output_dir"

for pdb_file in "$input_dir"/*.pdb; do
    base_name=$(basename "$pdb_file")
    output_name="${output_dir}/${base_name%.pdb}_H.pdb"

    echo "Processing $base_name..."

    gmx pdb2gmx -f "$pdb_file" -o "$output_name" -ignh -ff amber99sb -water none -ss no << EOF
1
EOF

    # Remove temporary or backup files starting with #
    find . -maxdepth 1 -type f -name '#*' -delete

done

