import torch
import os
import math
import sys

    # Set ranges and bins
xr = [0.1, 3.5]  # x-axis range (CV1)
yr = [0.1, 3.5]  # y-axis range (CV2)
zr = [0.1, 3.5]  # z-axis range (CV3)
nx, ny, nz = 100, 100, 100  # Number of bins for each dimension
dx = (xr[1] - xr[0]) / nx
dy = (yr[1] - yr[0]) / ny
dz = (zr[1] - zr[0]) / nz
    # Apply smoothing
sigma = 0.3  # Standard deviation of Gaussian kernel
    #logP_tensor = smooth_tensor(logP_tensor, sigma=sigma)
epsilon=1e-5
    # Define grid points
x = torch.linspace(xr[0], xr[1], nx)
y = torch.linspace(yr[0], yr[1], ny)
z = torch.linspace(zr[0], zr[1], nz)

def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel using PyTorch.
    """
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()  # Normalize
    return kernel

def smooth_tensor(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Smooth a 3D tensor using a Gaussian kernel.
    """
    # Create 1D Gaussian kernel
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)  # Ensure sufficient coverage
    kernel = gaussian_kernel(kernel_size, sigma)

    if tensor.ndim == 3:  # For 3D tensors
        kernel_nd = kernel.view(1, 1, -1) * kernel.view(1, -1, 1) * kernel.view(-1, 1, 1)
    elif tensor.ndim == 2:  # For 2D tensors
        kernel_nd = kernel.view(1, -1) * kernel.view(-1, 1)
    else:
        raise ValueError("Tensor dimensionality not supported for smoothing.")

    kernel_nd = kernel_nd / kernel_nd.sum()  # Normalize the kernel

    # Expand dimensions for convolution
    kernel_nd = kernel_nd.to(tensor.device)
    kernel_nd = kernel_nd.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, *kernel_shape]

    # Pad tensor to handle edges
    padding = kernel_size // 2
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    padding_args = (padding, padding, padding) if tensor.ndim == 5 else (padding, padding)
    tensor_padded = torch.nn.functional.pad(tensor, padding_args, mode="replicate")

    # Convolve with Gaussian kernel
    conv_fn = torch.nn.functional.conv3d if tensor.ndim == 5 else torch.nn.functional.conv2d
    smoothed_tensor = conv_fn(tensor_padded, kernel_nd)
    return smoothed_tensor.squeeze()

def interpolate_logP(coords, logP_tensor):
    """
    Interpolates logP using bilinear or trilinear interpolation with PyTorch.
    Handles both 2D and 3D cases dynamically.
    """
    #print(coords.shape)
    dim = logP_tensor.ndim  # Determine 2D or 3D based on logP_tensor dimensionality
    #print(dim)
    # Convert coordinates to PyTorch tensor if not already
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)
    coords = coords.to(logP_tensor.device)  # Ensure compatibility with logP_tensor device
    # Normalize coordinates to [0, 1]
    ranges = [xr, yr] if dim == 2 else [xr, yr, zr]
    coords_normalized = (coords - torch.tensor([r[0] for r in ranges], dtype=torch.float32)) / torch.tensor(
        [r[1] - r[0] for r in ranges], dtype=torch.float64
    )

    # Scale to [-1, 1] for PyTorch grid_sample
    coords_normalized = 2.0 * coords_normalized - 1.0

    # Handle out-of-bounds coordinates by clamping
    coords_normalized = torch.clamp(coords_normalized, -1.0, 1.0)

    # Reshape for grid_sample compatibility
    if dim == 2:
        coords_normalized = coords_normalized.view(1, 1, -1, 2)  # Shape for 2D grid
        logP_tensor = logP_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
    elif dim == 3:
        coords_normalized = coords_normalized.view(1, 1, -1, 1, 3)  # Shape for 3D grid
        logP_tensor = logP_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, depth, height, width]
    else:
        raise ValueError("logP_tensor must be 2D or 3D.")

    # Perform interpolation using grid_sample
    logP_values = torch.nn.functional.grid_sample(
        logP_tensor,
        coords_normalized,
        mode="bilinear",
        align_corners=True,
    ).squeeze()

    return logP_values

def finite_difference_gradient(data, logP_tensor, epsilon=1e-5):
    """
    Computes gradients using finite difference approximation.
    
    Parameters:
        data: Input coordinates tensor, shape (N, 2) for 2D or (N, 3) for 3D.
        logP_tensor: Tensor representing logP values.
        epsilon: Small perturbation for finite difference approximation.
    
    Returns:
        Gradients for each input coordinate.
    """
    gradients = torch.zeros_like(data, dtype=torch.float64, device=data.device)
    
    for i in range(data.shape[1]):  # Loop over each dimension
        # Perturb positively
        data_pos = data.clone()
        data_pos[:, i] += epsilon
        
        # Perturb negatively
        data_neg = data.clone()
        data_neg[:, i] -= epsilon
        
        # Interpolate values
        logP_pos = interpolate_logP(data_pos, logP_tensor)
        logP_neg = interpolate_logP(data_neg, logP_tensor)
        
        # Compute gradient via finite difference
        gradients[:, i] = (logP_pos - logP_neg) / (2 * epsilon)
    
    return gradients

def grad_log_golP_FD(data, res, sequence_to_file, sequence_list):
    # Set ranges and bins
    xr = [0.1, 3.5]  # x-axis range (CV1)
    yr = [0.1, 3.5]  # y-axis range (CV2)
    zr = [0.1, 3.5]  # z-axis range (CV3)
    nx, ny, nz = 100, 100, 100  # Number of bins for each dimension
    dx = (xr[1] - xr[0]) / nx
    dy = (yr[1] - yr[0]) / ny
    dz = (zr[1] - zr[0]) / nz
    # Apply smoothing
    sigma = 0.3  # Standard deviation of Gaussian kernel
    #logP_tensor = smooth_tensor(logP_tensor, sigma=sigma)
    epsilon=1e-5
    # Define grid points
    x = torch.linspace(xr[0], xr[1], nx)
    y = torch.linspace(yr[0], yr[1], ny)
    z = torch.linspace(zr[0], zr[1], nz)
    
    """
    Computes gradients of logP using finite difference approximation instead of autograd.
    """
    if data.device.type != 'cuda':
        raise ValueError(f"Data must be on GPU, but found on {data.device}")
    
    #if res == 0:
    #    data = data[:, 1:]  # Select appropriate subset
    #elif res == len(sequence_list) - 1:
    #    data = data[:, 0:2]

    # Load saved logP data from .pt file
    pot_name = sequence_to_file[sequence_list[res]]
    logP_tensor = torch.load(pot_name, weights_only=True).double()  # Safe loading

    # Compute gradients using finite difference
    gradients = finite_difference_gradient(data, logP_tensor, epsilon)
    
    #if res == 0:
    #    gradient = gradients[:, 0]
    #else:
    gradient = gradients[:, 1]

    return gradient
    