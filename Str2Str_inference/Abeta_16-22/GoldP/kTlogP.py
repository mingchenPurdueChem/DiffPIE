import numpy as np
import torch

def calc_fes_bin_3D(cv, weight, T):
    """
    Compute the Free Energy Surface (FES) for a 3D case.
    """
    hist = np.zeros((nx, ny, nz))
    for i in range(cv.shape[0]):
        ind_x = int((cv[i, 0] - xr[0]) / dx)
        ind_y = int((cv[i, 1] - yr[0]) / dy)
        ind_z = int((cv[i, 2] - zr[0]) / dz)
        if 0 <= ind_x < nx and 0 <= ind_y < ny and 0 <= ind_z < nz:
            hist[ind_x, ind_y, ind_z] += weight[i]
    hist = hist + 1.0e-10  # Prevent log(0)
    fes = -T * np.log(hist)
    fes -= np.min(fes)  # Normalize
    return fes

def calc_fes_bin_1D(cv, weight, T):
    hist = np.zeros(nx)
    for i in range(cv.shape[0]):
        ind_x = int((cv[i] - xr[0]) / dx)
        if 0 <= ind_x < nx:
            hist[ind_x] += weight[i]
    hist = hist + 1.0e-10  # Prevent log(0)
    fes = -T * np.log(hist)
    fes -= np.min(fes)  # Normalize
    return fes

# Load data
w = np.loadtxt("weight-out-only")
coord = np.loadtxt("CORD")
coord_data = np.column_stack((coord[:, 3], coord[:, 6], coord[:, 9]))
coord_data = coord_data-0.1
coord_data = coord_data[:len(w), :]

box = max_value = np.max(coord_data) - np.min(coord_data)
coord_data[coord_data < 0] += box
data = coord_data
# Set ranges and bins
xr = [0.1, 3.5]  # x-axis range (CV1)
yr = [0.1, 3.5]  # y-axis range (CV2)
zr = [0.1, 3.5]  # z-axis range (CV3)
nx, ny, nz = 100, 100, 100  # Number of bins for each dimension
dx = (xr[1] - xr[0]) / nx
dy = (yr[1] - yr[0]) / ny
dz = (zr[1] - zr[0]) / nz

T = 300.0 * 8.314e-3 / 4.182  # Temperature in k_B units

FES = calc_fes_bin_3D(data[:len(w), :], w, T)
logP = -FES/T
logP_tensor = torch.tensor(logP, dtype=torch.float64)
torch.save(logP_tensor, "logP.pt")

data_1D = coord_data[:len(w), 1]  # Ca
FES = calc_fes_bin_1D(data_1D, w, T)
logP = -FES/T
logP_tensor = torch.tensor(logP, dtype=torch.float64)
torch.save(logP_tensor, "logP1D.pt")


