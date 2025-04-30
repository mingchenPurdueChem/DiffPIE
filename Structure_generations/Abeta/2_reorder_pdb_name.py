import os
import re

folder = 'individual_AA'
pattern = re.compile(r'^(.*_\d+)_([0-9]+)_([A-Za-z]+)\.pdb$')

for filename in os.listdir(folder):
    if filename.endswith('.pdb'):
        match = pattern.match(filename)
        if match:
            base, part1, part2 = match.groups()
            new_filename = f"{base}_{part2}_{part1}.pdb"
            src = os.path.join(folder, filename)
            dst = os.path.join(folder, new_filename)
            os.rename(src, dst)
            print(f"Renamed: {filename} → {new_filename}")

