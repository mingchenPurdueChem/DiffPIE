input_pdb = "Str2Str/logs/inference/runs/Au_F_2/samples/all_delta/GKLVFFAEG_F_2.pdb"
output_pdb = input_pdb.replace(".pdb", "_noGLY.pdb")

with open(input_pdb, "r") as infile, open(output_pdb, "w") as outfile:
    for line in infile:
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "GLY":
            continue
        outfile.write(line)

print(f"Saved to: {output_pdb}")
