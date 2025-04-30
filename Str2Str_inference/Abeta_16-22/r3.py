"""Inspired by https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py"""

from math import sqrt
import math
import torch
import subprocess
from src.utils.tensor_utils import inflate_array_like
import json
import joblib
import numpy as np
from scipy.optimize import approx_fprime
import os
from Abeta_biasing import grad_log_golP_FD
import pickle

class R3Diffuser:
    """VPSDE diffusion module."""
    def __init__(
        self,
        min_b: float = 0.1,
        max_b: float = 20.0,
        coordinate_scaling: float = 1.0,
    ):
        self.min_b = min_b
        self.max_b = max_b
        self.coordinate_scaling = coordinate_scaling
        #print(self.coordinate_scaling)
        #print(self.max_b)

    def scale(self, x):
        return x * self.coordinate_scaling

    def unscale(self, x):
        return x / self.coordinate_scaling

    def b_t(self, t: torch.Tensor):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        return -0.5 * self.b_t(t) * x

    def sample_prior(self, shape, device=None):
        return torch.randn(size=shape, device=device)

    def marginal_b_t(self, t):
        return t*self.min_b + 0.5*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (score_t * cond_var + x_t) / torch.exp(-0.5*beta_t)

    def forward_marginal(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        t = inflate_array_like(t, x_0)
        x_0 = self.scale(x_0)

        loc = torch.exp(-0.5 * self.marginal_b_t(t)) * x_0
        scale = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))
        z = torch.randn_like(x_0)
        x_t = z * scale + loc
        score_t = self.score(x_t, x_0, t)

        x_t = self.unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.Tensor):
        return 1.0 / torch.sqrt(self.conditional_var(t))

    def trans_hat(
        self,
        x_t: torch.Tensor,
        score_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
        center: bool = True,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
        N=1,
        T=1, # T is delta 0-1
        ts=None
    ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.
            probability_flow: whether to use probability flow ODE.

        Returns:
            [..., 3] positions at next step t-1.
        """
        ts_reversed = ts[::-1].copy()
        ts = torch.tensor(ts_reversed, dtype=torch.float32)
        discrete_betas = self.b_t(ts) / N
        alphas = 1.0 - discrete_betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        x_t = self.scale(x_t)
        timestep = (t * (N - 1) / T).long()
        alpha_t = alphas_cumprod.to(x_t.device)[timestep]
        one_minus_alpha_t = 1 - alpha_t
        # Compute x_hat
        x_hat = (x_t + one_minus_alpha_t[:, None, None] * score_t) / torch.sqrt(alpha_t[:, None, None])
        J_hat = 1/torch.sqrt(alpha_t[:, None, None]) #N,1,1 easy for boradcasting
        return x_hat,J_hat

    def reverse(
        self,
        x_hat: torch.Tensor,
        J_hat: torch.Tensor,
        x_t: torch.Tensor,
        score_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
        center: bool = True,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
        target_distance: torch.Tensor = torch.tensor([0]), # need to scale by 0.1 so in nm
        target_pair_linker: torch.Tensor = torch.tensor([[0, 7],[1, 6],[1, 8]]),
        k_spring: float = 10
    ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.
            probability_flow: whether to use probability flow ODE.

        Returns:
            [..., 3] positions at next step t-1.
        """
        t_current = t
        ######################### force scaling
        m = torch.tensor(1.5)
        force_scale = 1-(torch.exp(m * t_current[0]) - 1) / (torch.exp(m) - 1)
        force_scale = force_scale.to(torch.float32)
        
        t = inflate_array_like(t, x_t)
        x_t = self.scale(x_t)

        f_t = self.drift_coef(x_t, t)
        g_t = self.diffusion_coef(t)

        z = noise_scale * torch.randn_like(score_t)
        drift_adjustment = torch.zeros_like(x_t)
        
        ######################################################
        #print(x_hat.device)
        #print(x_t.device)
        #trans_rep = torch.zeros_like(x_t)
        #trans_inter = torch.zeros_like(x_t)
        #with torch.no_grad():
        #    x_hat = x_hat.detach().clone().requires_grad_(True)
            #x_hat.requires_grad = True
            #print(f"x_hat grad_fn (before computation): {x_hat.grad_fn}")
        #    pot_dir = './GoldP'
        #    pt_files = [f for f in os.listdir(pot_dir) if f.endswith('_logP.pt')]
        #    full_sequence = 'GKLVFFAEG'
        #    full_sequence_list = list(full_sequence)
        #    sequence = 'KLVFFAE'
        #    sequence_list = list(sequence)
        
            # Create a mapping dictionary for the sequence
        #    sequence_to_file = {}
        #    for i in range(len(sequence_list)):
        #        file_name = f"{sequence_list[i]}_logP3.pt"
        #        sequence_to_file[sequence_list[i]] = os.path.join(pot_dir, file_name)
                
            #torch.save(sequence_to_file, "sequence_to_file.pt")
            
        #    k_rep = 10
        #    Au_Z = 0.2  # Au at Z = xx nm
        #    cutoff = 0.1
        #    Au_Z_cutoff = Au_Z - cutoff
        
            # Identify indices where x_hat[..., 2] > Au_Z_cutoff
        #    beyond_mask = x_hat[..., 2] > Au_Z_cutoff
        #    dis = Au_Z_cutoff - x_hat[..., 2]
            #print(dis[...,0])
            #print(beyond_mask)
            #print(beyond_mask.shape)
        #    beyond_indices = torch.nonzero(beyond_mask, as_tuple=False)
            # Get indices where the mask is True
            
            #print("Beyond mask:", beyond_mask)
            #print("Beyond mask indices:", beyond_indices)# Get indices where the mask is True
            #beyond_dis = x_hat[beyond_indices[:, 0], beyond_indices[:, 1], 2] - Au_Z_cutoff
            
            #print(f"beyond_dis grad_fn (before computation): {beyond_dis.grad_fn}")
        
            # Compute spring force
            #spring_force = -k_rep * beyond_dis
            #trans_rep[beyond_indices[:, 0], beyond_indices[:, 1], 2] += spring_force
        #    if beyond_indices.numel() > 0:  # Check if there are valid indices
        #        for idx in range(beyond_indices.size(0)):  # Iterate over each valid index
        #            i, j = beyond_indices[idx]  # Extract the current indices
        #            beyond_dis = x_hat[i, j, 2] - Au_Z_cutoff  # Compute the displacement beyond the cutoff
        #            spring_force = -k_rep * beyond_dis  # Calculate the spring force
        #            trans_rep[i, j, 2] += spring_force  # Apply the force in the z-direction
        
            # Iterate over residues
        #    for full_res in range(x_hat.shape[1]-2):#two G
        #        # Compute displacements for current, previous, and next residues
        #        res = full_res + 1
        #        cood_1 = Au_Z - x_hat[:, res, 2]  # Shape: (N,)
        #        codd_0 = (Au_Z - x_hat[:, res - 1, 2]) if res > 0 else torch.zeros_like(cood_1)
        #        codd_2 = (Au_Z - x_hat[:, res + 1, 2]) if res < x_hat.shape[1] - 1 else torch.zeros_like(cood_1)
        
                # Stack displacements into a single tensor
        #        codd = torch.stack([codd_0, cood_1, codd_2], dim=-1)  # Shape: (N, 3)
                
                #with torch.enable_grad():
        #        grad_logP = grad_log_golP_FD(codd, full_res, sequence_to_file, sequence_list)
                    #if grad_logP.grad is not None:
                     #   grad_logP.grad.zero_()
                #if not torch.isnan(codd).any() and timestep % 100 == 0:
                #if not torch.isnan(codd).any():
                    #print("Warning: codd contains NaN values!")
                    #print(codd)
                    #print(grad_logP*force_scale)
                    #print(timestep)
                    #break
                
        #       truncated_force = torch.clamp(grad_logP * force_scale, min=-3, max=3)
                #print(grad_logP.shape)
                #truncated_force = truncated_force.detach().clone().requires_grad_(False)
        #       trans_inter[..., res, 2] = -truncated_force # Positive grad_logP means elongate distance
                # Compute gradient of logP
                #grad_logP = grad_log_golP(codd, res, sequence_to_file, sequence_list)
        
                # Update trans_inter for the current residue
        
            # Reset trans_inter for beyond indices
            #trans_inter[beyond_indices[:, 0], beyond_indices[:, 1], 2] = 0
        
            # Compute the final trans_drift
            #trans_drift = trans_inter + trans_rep
            #print(trans_inter[...,0,2])
            #print(trans_rep[...,0,2])
            #print(score_t[...,0,2])
            #trans_drift = trans_inter + trans_rep
            
                #for idx, force in zip(beyond_indices, spring_force):
                #    trans_inter[idx[0], idx[1], 2] = 0  # remove interaction for beyond        
        
        #############################################
        # Step 1: Calculate end-to-end vectors, distances, and unit vectors
        #gmm_potential_model.pkl
        idx1, idx2 = target_pair_linker[:, 0], target_pair_linker[:, 1]
        end_to_end_vector = x_hat[..., idx1, :] - x_hat[..., idx2, :]
        end_to_end_vector = end_to_end_vector
        current_distance = torch.norm(end_to_end_vector, dim=-1, keepdim=True)
        unit_vector = end_to_end_vector / (current_distance + 1e-8)

        # Step 2: Convert distances to NumPy
        distances_np = current_distance.cpu().numpy().squeeze(-1)
        distances_np[np.isnan(distances_np)] = 0
        #print(distances_np[0:5])

        # Step 3: Load the KDE model and calculate gradient of logP using finite differences
        #kde = joblib.load('kde_model_empirical.pkl')
        #epsilon = np.sqrt(np.finfo(float).eps)       
        
        #def log_prob(x):
        #    return kde.score_samples(x.reshape(1, -1))[0]

        #batch_grad_logP = np.array([approx_fprime(x, log_prob, epsilon) for x in distances_np])

        # Step 4: Convert gradient back to PyTorch tensor on GPU
        #batch_grad_logP_tensor = torch.tensor(batch_grad_logP, dtype=torch.float64).cuda()
        #gmm = pickle.load('gmm_potential_model.pkl')
        #epsilon = np.sqrt(np.finfo(float).eps)
        # Open the file in binary read mode
        with open('gmm_potential_model.pkl', 'rb') as file:
            saved_data = pickle.load(file)  # Load the dictionary
        gmm = saved_data['gmm_model']  # Now gmm is a GaussianMixture instance
        
        def grad_log_prob_gmm(x):
            """ Compute the gradient of log probability for a GMM analytically. """
            x = x.reshape(1, -1)  # Ensure shape is (1, N)
    
            # Responsibilities (probabilities of each component given x)
            gamma = gmm.predict_proba(x)  # Shape: (1, n_components)
            #gamma = gmm.score_samples(x.reshape(1, -1))[0]

            # Mean and covariance matrices of each Gaussian component
            means = gmm.means_  # Shape: (n_components, N)
            covariances = gmm.covariances_  # Shape: (n_components, N, N)

            # Compute inverse of covariance matrices
            cov_inv = np.linalg.inv(covariances)  # Shape: (n_components, N, N)

            # Compute the gradient
            grad = np.sum(
                      gamma[:, :, None] * np.einsum('kij,kj->ki', cov_inv, (means - x)), axis=1
            )  # Shape: (1, N)

            return grad.squeeze(0)  # Shape: (N,)

        # Compute batch gradients
        batch_grad_logP = np.array([grad_log_prob_gmm(x) for x in distances_np])

        # Convert to PyTorch tensor and move to GPU
        batch_grad_logP_tensor = torch.tensor(batch_grad_logP, dtype=torch.float64).cuda()       

        #print(batch_grad_logP_tensor[0])
        
        spring_force = batch_grad_logP_tensor.unsqueeze(-1) * unit_vector *force_scale*0.1
        # Step 5: Apply spring force adjustment
        for i in range(batch_grad_logP_tensor.shape[-1]):
            
            #spring_force = batch_grad_logP_tensor[:, i] * unit_vector[:, i ,:]
            #spring_force = batch_grad_logP_tensor[:, i, None] * unit_vector[:, i, :]
            drift_adjustment[..., target_pair_linker[i, 0], :] += spring_force[:, i, :]
            drift_adjustment[..., target_pair_linker[i, 1], :] -= spring_force[:, i, :]
        
        
        ##############################################
        # constant force
        #vector = x_hat[..., [-1], :]-x_hat[..., [0], :]
        #epsilon = 1e-8
        #norm_vector = vector / (torch.norm(vector, dim=-1, keepdim=True) + epsilon)
        # 1pN = 1/69.5 KCal/mol/A
        # 1/KbT = 1.667 mol/KCal
        # grad_logP = pN/KbT = 0.024 1/A
        #convertion = 0.024/self.coordinate_scaling
        #force = 0
        #constant_force = force*convertion
        #trans_drift[..., [-1], :] += norm_vector*constant_force  # Apply equal and opposite force to the second position
        #trans_drift[..., [0], :] -= norm_vector*constant_force
        
        ##############################################
        # here to ensure the Ca-Ca is 3.8A
        N = x_t.shape[1]
        target_pair = torch.stack([torch.arange(N - 1), torch.arange(1, N)], dim=1).to('cuda')
        target_distance = torch.tensor([0.38] * (N - 1)).to('cuda')
        
        # Extract indices from target_pair
        idx1, idx2 = target_pair[:, 0], target_pair[:, 1]

        # Calculate end-to-end vectors and distances
        end_to_end_vector = x_hat[..., idx1, :] - x_hat[..., idx2, :]
        current_distance = torch.norm(end_to_end_vector, dim=-1, keepdim=True)

        # Calculate distance errors
        distance_error = current_distance - target_distance.unsqueeze(-1)

        # Calculate unit vectors
        unit_vector = end_to_end_vector / (current_distance + 1e-8)

        # Calculate spring forces
        spring_force = -k_spring * distance_error * unit_vector

        # Apply the forces
        drift_adjustment[..., idx1, :] += spring_force
        drift_adjustment[..., idx2, :] -= spring_force

        trans_drift = (drift_adjustment)*J_hat
        
        #trans_drift = torch.zeros_like(x_t)

        rev_drift = (f_t - g_t ** 2 * (score_t+ trans_drift)) * dt * (0.5 if probability_flow else 1.)
        rev_diffusion = 0. if probability_flow else (g_t * sqrt(dt) * z)
        perturb = rev_drift + rev_diffusion

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones_like(x_t[..., 0])
        x_t_1 = x_t - perturb   # reverse in time
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1)[..., None] # reduce length dim
            x_t_1 -= com[..., None, :]

        x_t_1 = self.unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).
        Var[x_t|x_0] = conditional_var(t) * I
        """
        return 1.0 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, scale=False):
        t = inflate_array_like(t, x_t)
        if scale:
            x_t, x_0 = self.scale(x_t), self.scale(x_0)
        return -(x_t - torch.exp(-0.5 * self.marginal_b_t(t)) * x_0) / self.conditional_var(t)

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self.scale(x_t)
        f_t = self.drift_coef(x_t, t)
        g_t = self.diffusion_coef(t)
        std = g_t * sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std
