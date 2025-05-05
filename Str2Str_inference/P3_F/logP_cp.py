import mdtraj as md
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
#######################
trajectory = md.load('IZS.gro')
data = trajectory.xyz
#######################
# Cn-1 = 21 Cn = 23 Cn+1 = 29
# Cm-1 = 40 Cm = 34 Cm+1 = 32
# pairs 21-40 23-34 29-32 
# pairs 21-34 23-32 29-40
# Compute Euclidean distances (L2 norm) for all six pairs
d1 = np.linalg.norm(data[:, 20, :] - data[:, 39, :], axis=1)  # 21-40
d2 = np.linalg.norm(data[:, 22, :] - data[:, 33, :], axis=1)  # 23-34
d3 = np.linalg.norm(data[:, 28, :] - data[:, 31, :], axis=1)  # 29-32
d4 = np.linalg.norm(data[:, 20, :] - data[:, 33, :], axis=1)  # 21-34
d5 = np.linalg.norm(data[:, 22, :] - data[:, 31, :], axis=1)  # 23-32
d6 = np.linalg.norm(data[:, 28, :] - data[:, 39, :], axis=1)  # 29-40
# Combine into an array if needed
all_distances = np.stack([d1, d2, d3, d4, d5, d6], axis=1)
#######################
# Build a GMM model
num_components = 10  # Adjust as needed
gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
gmm.fit(all_distances)

# Save the GMM model and associated data
with open('gmm_potential_model.pkl', 'wb') as file:
    pickle.dump({'gmm_model': gmm, 'data_points': all_distances}, file)
print("GMM model and data points saved as 'gmm_potential_model.pkl'")