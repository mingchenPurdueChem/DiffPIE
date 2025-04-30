#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:49:58 2025

@author: yanbin
"""

import numpy as np
from scipy.spatial import cKDTree

# Load Ca and S coordinates for Str2Str (reference)
ca_data = np.load('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/Ca10_local_coordinates_str2str.npy')
s_data = np.load('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/s10_local_coordinates_str2str.npy')

# Load Ca and S coordinates for Simulation (model)
ca_model = np.load('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/MD/Ca10_local_coordinates_simulation.npy')
s_model = np.load('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/MD/s10_local_coordinates_simulation.npy')

# Check shapes to make sure they match
assert ca_data.shape == s_data.shape, "Mismatch between Ca and S data shapes!"
assert ca_model.shape == s_model.shape, "Mismatch between Ca and S model shapes!"

#print(f"Data shape (Ca and S): {ca_data.shape}")
#print(f"Model shape (Ca and S): {ca_model.shape}")

# Combine Ca and S coordinates into a single feature vector per frame (6D: x, y, z for Ca and S)
combined_data = np.hstack([ca_data, s_data])      # Shape (n_frames, 6)
combined_model = np.hstack([ca_model, s_model])   # Shape (n_frames, 6)

#print(combined_model[221254])
#print(combined_data[38])

#print(f"Combined Data shape: {combined_data.shape}")
#print(f"Combined Model shape: {combined_model.shape}")

# Build KDTree using the combined coordinates
tree = cKDTree(combined_model)

# Query the 10 nearest neighbors for each combined data point
distances, indices = tree.query(combined_data, k=10)

# Print the 10 most similar indices for the first 10 data points
#print("\nTop 10 closest model indices for each of the first 10 data points:")

for i in range(min(10, len(combined_data))):
    print(f"\nData point {i}:")
    print(f"Coordinates (Ca + S): {combined_data[i]}")
    print(f"Closest model indices: {indices[i]}")
    print(f"Distances: {distances[i]}")
    print(f"Closest model points (Ca + S):\n{combined_model[indices[i]]}")

# Optional: save all indices and distances
np.save('/home/yanbin/Desktop/Projects/organic_linker/cyclic_linker_2.0/data/biased/2/top10_closest_model_indices.npy', indices)
#np.save('top10_distances.npy', distances)

# Example: print a specific model point and data point for verification
print("\nExample: Model point at index 5580 (Ca + S):", combined_model[5580])
print("Example: Data point 1 (Ca + S):", combined_data[0])
