import os
from typing import Any, Dict, Tuple, Optional
from random import random
from copy import deepcopy

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.models.score.frame import FrameDiffuser
from src.models.loss import ScoreMatchingLoss
from src.common.rigid_utils import Rigid
from src.common.all_atom import compute_backbone
from src.common.pdb_utils import atom37_to_pdb, merge_pdbfiles
from src.common import rotation3d
from src.common.rigid_utils import Rigid, Rotation, quat_multiply

def assemble_rigid(rotvec: torch.Tensor, trans: torch.Tensor):
    trans_t_1_assemble = trans
    neighboring_distances = torch.norm(trans_t_1_assemble[:, 1:, :] - trans_t_1_assemble[:, :-1, :], dim=2)
    #print("neighboring distance in assemble",neighboring_distances[0])
    rotvec_shape = rotvec.shape
    rotmat = rotation3d.axis_angle_to_matrix(rotvec).view(rotvec_shape[:-1] + (3, 3))
    return Rigid(
        rots=Rotation(rot_mats=rotmat),
        trans=trans,
    )

def adjust_ca_distances(batch_data, target_distance=3.8, max_distance=10.0):
    """
    Adjust the distances between consecutive Cα atoms in each batch to the target distance,
    ignoring distances larger than max_distance.

    Parameters:
    - batch_data: torch.Tensor of shape (N_batch, N_ca, 3) representing the coordinates.
    - target_distance: float, the target distance between consecutive Cα atoms.
    - max_distance: float, the maximum distance to consider for adjustment.

    Returns:
    - adjusted_batch_data: torch.Tensor of shape (N_batch, N_ca, 3) with adjusted coordinates.
    """
    N_batch, N_ca, _ = batch_data.shape
    adjusted_batch_data = batch_data.clone()

    for i in range(1, N_ca):
        # Calculate the vector from the previous Cα atom to the current Cα atom
        vector = adjusted_batch_data[:, i, :] - adjusted_batch_data[:, i - 1, :]
        # Calculate the current distance
        distance = torch.norm(vector, dim=1, keepdim=True)
        # Mask to exclude distances larger than max_distance
        mask = distance <= max_distance
        # Calculate the scaling factor only for valid distances
        scale = torch.where(mask, target_distance / distance, torch.ones_like(distance))
        # Adjust the current position based on the scaled vector
        adjustment = vector * (scale - 1)
        adjusted_batch_data[:, i, :] += adjustment

    return adjusted_batch_data


class DiffusionLitModule(LightningModule):
    """Example of a `LightningModule` for denoising diffusion training.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        diffuser: FrameDiffuser,
        loss: Dict[str, Any],
        compile: bool,
        inference: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network and diffusion module
        self.net = net
        self.diffuser = diffuser

        # loss function
        self.loss = ScoreMatchingLoss(config=self.hparams.loss)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.
        (Not actually used)

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # preprocess by augmenting additional feats
        rigids_0 = Rigid.from_tensor_4x4(batch['rigidgroups_gt_frames'][..., 0, :, :])
        batch_size = rigids_0.shape[0]
        t = (1.0 - self.diffuser.min_t) * torch.rand(batch_size, device=rigids_0.device) + self.diffuser.min_t
        perturb_feats = self.diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=None,
            as_tensor_7=True,
        )
        patch_feats = {
            't': t,
            'rigids_0': rigids_0,
        }
        batch.update({**perturb_feats, **patch_feats})
        
        use_grad = True
        # probably add self-conditioning (recycle once)
        if self.net.embedder.self_conditioning and random() > 0.5:
            #with torch.no_grad():# here we might need to turn off
            if use_grad:
            #with torch.enable_grad():
                batch['sc_ca_t'] = self.net(batch, as_tensor_7=True)['rigids'][..., 4:]

        # feedforward
        out = self.net(batch)

        # postprocess by add score computation
        pred_scores = self.diffuser.score(
            rigids_0=out['rigids'],
            rigids_t=Rigid.from_tensor_7(batch['rigids_t']),
            t=t,
            mask=batch['residue_mask'],
        )
        out.update(pred_scores)

        # calculate losses
        loss, loss_bd = self.loss(out, batch, _return_breakdown=True)
        return loss, loss_bd

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_bd = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k,v in loss_bd.items():
            if k == 'loss': continue
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd = self.model_step(batch, training=False)

        # update and log metrics
        self.val_loss(loss) # update
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        _vall = self.val_loss.compute()  # get current val acc
        self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError("Test step not implemented.")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> str:
        """Perform a prediction step on a batch of data from the dataloader.
    
        This prediction step will sample `n_replica` copies from the forward-backward process,
        repeated for each delta-T in the range of [delta_min, delta_max] with step size
        `delta_step`. If `backward_only` is set to True, then only the backward process will be
        performed, and `n_replica` will be multiplied by the number of delta-Ts.
    
        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        #is_inference_mode = torch.is_inference_mode_enabled()
        #print(f"Inference Mode Active: {is_inference_mode}")
        with torch.no_grad():
            
             #with torch.enable_grad():
            #x = torch.tensor([0.0, 1.0, 1.0], device='cuda:0', requires_grad=True)   
            #y = x ** 2
            #print(x.device)
            #print(f"Is x a leaf tensor? {x.is_leaf}")
            #print(f"y.grad_fn: {y.grad_fn}") 
            
            # Extract hyperparameters for inference
            n_replica = self.hparams.inference.n_replica
            replica_per_batch = self.hparams.inference.replica_per_batch
            delta_range = np.arange(
                self.hparams.inference.delta_min,
                self.hparams.inference.delta_max + 1e-5,
                self.hparams.inference.delta_step
            )
            delta_range = np.around(delta_range, decimals=2)  # Up to 2 decimal places
            num_timesteps = self.hparams.inference.num_timesteps
            noise_scale = self.hparams.inference.noise_scale
            probability_flow = self.hparams.inference.probability_flow
            self_conditioning = self.hparams.inference.self_conditioning and self.net.embedder.self_conditioning
            min_t = self.hparams.inference.min_t
            output_dir = self.hparams.inference.output_dir
            backward_only = self.hparams.inference.backward_only
    
            # If backward_only, perform only the backward process (vanilla sampling of diffusion)
            if backward_only:
                n_replica *= len(delta_range)
                delta_range = [-1.0]
    
            assert batch['aatype'].shape[0] == 1, "Batch size must be 1 for correct inference."
    
            # Get extra features of the current protein
            accession_code = batch['accession_code'][0]
            extra = {
                'aatype': batch['aatype'][0].detach().cpu().numpy(),
                'chain_index': batch['chain_index'][0].detach().cpu().numpy(),
                'residue_index': batch['residue_index'][0].detach().cpu().numpy(),
            }
    
            # Define sampling subroutine
            def forward_backward(rigids_0: Rigid, t_delta: float):
                T = t_delta if t_delta > 0 else 1.0
                batch_size, device = rigids_0.shape[0], rigids_0.device
                _num_timesteps = int(float(num_timesteps) * T)
                dt = 1.0 / _num_timesteps
                ts = np.linspace(min_t, T, _num_timesteps)[::-1]  # Reverse in time
    
                _feats = deepcopy({
                    k: v.repeat(batch_size, *(1,) * (v.ndim - 1))
                    for k, v in batch.items() if k in (
                        'aatype', 'residue_mask', 'fixed_mask',
                        'residue_idx', 'torsion_angles_sin_cos'
                    )
                })
    
                _feats['fixed_mask'][..., 0:3] = 0
                diffuse_mask = (1 - _feats['fixed_mask']) * _feats['residue_mask']
    
                if t_delta > 0:
                    rigids_t = self.diffuser.forward_marginal(
                        rigids_0=rigids_0,
                        t=t_delta * torch.ones(batch_size, device=device),
                        diffuse_mask=_feats['residue_mask'],
                        as_tensor_7=True,
                    )['rigids_t']
                else:
                    rigids_t = self.diffuser.sample_prior(
                        shape=rigids_0.shape,
                        device=device,
                        reference_rigids=rigids_0,
                        diffuse_mask=diffuse_mask,
                        as_tensor_7=True,
                    )['rigids_t']
    
                _feats['rigids_t'] = rigids_t
    
                traj_atom37 = []
                use_grad = True
                if use_grad:
                    fixed_mask = _feats['fixed_mask'] * _feats['residue_mask']
                    diffuse_mask = (1 - _feats['fixed_mask']) * _feats['residue_mask']
    
                    if self_conditioning:
                        _feats['sc_ca_t'] = torch.zeros_like(rigids_t[..., 4:])
                        _feats['t'] = ts[0] * torch.ones(batch_size, device=device)
                        _feats['sc_ca_t'] = self.net(_feats, as_tensor_7=True)['rigids'][..., 4:]
    
                    temp_set_grad_flag = False
                    for t in ts:
                        _feats['t'] = t * torch.ones(batch_size, device=device)
                    
                        if temp_set_grad_flag:
                            with torch.enable_grad():
                                 out = self.net(_feats, as_tensor_7=False)
                        else:
                            with torch.no_grad():
                                 out = self.net(_feats, as_tensor_7=False)
                        #out = self.net(_feats, as_tensor_7=False)
                        if t == min_t:
                            rigids_pred = out['rigids']
                            #rigids_pred_trans = rigids_pred.get_trans()
                            #rigids_pred_rot = rotation3d.matrix_to_axis_angle(rigids_pred.get_rots().get_rot_mats())
                            #rigids_pred_trans = adjust_ca_distances(rigids_pred_trans)
                            #rigids_pred = assemble_rigid(rigids_pred_rot, rigids_pred_trans)
                        else:
                            if self_conditioning:
                                _feats['sc_ca_t'] = out['rigids'].to_tensor_7()[..., 4:]
                            pred_scores = self.diffuser.score(
                                rigids_0=out['rigids'],
                                rigids_t=Rigid.from_tensor_7(_feats['rigids_t']),
                                t=_feats['t'],
                                mask=_feats['residue_mask'],
                            )
                            rigids_pred = self.diffuser.reverse(
                                rigids_t=Rigid.from_tensor_7(_feats['rigids_t']),
                                rot_score=pred_scores['rot_score'],
                                trans_score=pred_scores['trans_score'],
                                t=_feats['t'],
                                dt=dt,
                                num_step=_num_timesteps,
                                T=T,
                                ts=ts,
                                diffuse_mask=diffuse_mask,
                                center_trans=True,
                                noise_scale=noise_scale,
                                probability_flow=probability_flow,
                                set_grad_flag=temp_set_grad_flag,
                            )
                            if temp_set_grad_flag:
                                temp_set_grad_flag = False
                            _feats['rigids_t'] = rigids_pred.to_tensor_7()
    
                    atom37 = compute_backbone(rigids_pred, out['psi'], aatype=_feats['aatype'])[0]
                    atom37 = atom37.detach().cpu().numpy()
                    return atom37
    
            saved_paths = []
    
            for t_delta in delta_range:
                gt_rigids_4x4 = batch['rigidgroups_gt_frames'][..., 0, :, :].clone()
                n_bs = n_replica // replica_per_batch
                last_bs = n_replica % replica_per_batch
                atom_positions = []
                for _ in range(n_bs):
                    rigids_0 = Rigid.from_tensor_4x4(
                        gt_rigids_4x4.repeat(replica_per_batch, *(1,) * (gt_rigids_4x4.ndim - 1))
                    )
                    traj_atom37 = forward_backward(rigids_0, t_delta)
                    atom_positions.append(traj_atom37)
                if last_bs > 0:
                    rigids_0 = Rigid.from_tensor_4x4(
                        gt_rigids_4x4.repeat(last_bs, *(1,) * (gt_rigids_4x4.ndim - 1))
                    )
                    traj_atom37 = forward_backward(rigids_0, t_delta)
                    atom_positions.append(traj_atom37)
                atom_positions = np.concatenate(atom_positions, axis=0)
    
                t_delta_dir = os.path.join(output_dir, f"{t_delta}")
                os.makedirs(t_delta_dir, exist_ok=True)
                save_to = os.path.join(t_delta_dir, f"{accession_code}.pdb")
                saved_to = atom37_to_pdb(
                    atom_positions=atom_positions,
                    save_to=save_to,
                    **extra,
                )
                saved_paths.append(saved_to)
    
            all_delta_dir = os.path.join(output_dir, "all_delta")
            os.makedirs(all_delta_dir, exist_ok=True)
            merge_pdbfiles(saved_paths, os.path.join(all_delta_dir, f"{accession_code}.pdb"))
    
            return all_delta_dir
    
    

if __name__ == "__main__":
    _ = DiffusionLitModule(None, None, None, None, None)
