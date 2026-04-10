"""
ChiralBoltzModule: PyTorch Lightning module that fine-tunes Boltz-2 with
the additional differentiable chiral volume loss term.

Architecture:
  - Loads pre-trained Boltz-2 weights (jwohlwend/boltz)
  - Adds chiralboltz.loss.chiral_volume.chiral_volume_loss as an auxiliary loss
  - Loss = L_boltz2 + lambda_chiral * L_chiral
  - lambda_chiral is linearly warmed up from 0 to target over warmup_steps
  - All other Boltz-2 training logic (diffusion loss, confidence loss) is inherited
"""

import torch
import pytorch_lightning as pl
from torch import Tensor

from chiralboltz.loss.chiral_volume import chiral_volume_loss, extract_chiral_centers_from_batch
from chiralboltz.training.config import ChiralBoltzConfig


class ChiralBoltzModule(pl.LightningModule):
    """
    Fine-tuning wrapper for Boltz-2 with chiral volume auxiliary loss.

    Parameters
    ----------
    boltz2_model : nn.Module
        A loaded Boltz-2 model (from boltz.model.models.boltz2.Boltz2).
    config : ChiralBoltzConfig
        Training configuration.
    """

    def __init__(self, boltz2_model, config: ChiralBoltzConfig):
        super().__init__()
        self.model  = boltz2_model
        self.config = config
        self.save_hyperparameters(ignore=['boltz2_model'])

    # ------------------------------------------------------------------
    # Core forward / loss
    # ------------------------------------------------------------------

    def _compute_chiral_loss(self, pred_coords: Tensor, true_coords: Tensor, batch: dict) -> Tensor:
        """Extract chiral centers and compute the chiral volume loss."""
        chiral_centers, chiral_mask = extract_chiral_centers_from_batch(batch)
        return chiral_volume_loss(
            pred_coords=pred_coords,
            true_coords=true_coords,
            chiral_centers=chiral_centers,
            chiral_mask=chiral_mask,
            mode=self.config.chiral_loss_mode,
            sign_margin=self.config.chiral_sign_margin,
            mse_weight=self.config.chiral_mse_weight,
        )

    def _chiral_lambda(self) -> float:
        """Linear warmup schedule for chiral loss weight."""
        step = self.global_step
        warmup = self.config.chiral_loss_warmup_steps
        if warmup <= 0:
            return self.config.lambda_chiral
        return self.config.lambda_chiral * min(1.0, step / warmup)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        # --- Boltz-2 base forward pass ---
        # Boltz-2's training_step returns a dict with 'loss' and 'pred_coords'
        # We call the model's loss computation directly.
        boltz2_out = self.model.training_step(batch, batch_idx)

        base_loss   = boltz2_out['loss']
        pred_coords = boltz2_out.get('pred_coords')   # Float[B, N, 3] — last diffusion sample
        true_coords = batch.get('atom_coords')         # Float[B, N, 3]

        # --- Chiral volume auxiliary loss ---
        chiral_loss = torch.zeros((), device=base_loss.device)
        if pred_coords is not None and true_coords is not None:
            chiral_loss = self._compute_chiral_loss(pred_coords, true_coords, batch)

        lam = self._chiral_lambda()
        total_loss = base_loss + lam * chiral_loss

        # --- Logging ---
        self.log('train/loss',        total_loss,  on_step=True, prog_bar=True)
        self.log('train/base_loss',   base_loss,   on_step=True)
        self.log('train/chiral_loss', chiral_loss, on_step=True)
        self.log('train/chiral_lambda', lam,       on_step=True)

        mirrored_frac = batch.get('is_mirrored', torch.zeros(1)).float().mean()
        self.log('train/mirrored_fraction', mirrored_frac, on_step=True)

        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        boltz2_out = self.model.validation_step(batch, batch_idx)
        base_loss  = boltz2_out['loss']

        pred_coords = boltz2_out.get('pred_coords')
        true_coords = batch.get('atom_coords')

        chiral_loss = torch.zeros((), device=base_loss.device)
        if pred_coords is not None and true_coords is not None:
            chiral_loss = self._compute_chiral_loss(pred_coords, true_coords, batch)

        lam = self._chiral_lambda()
        total_loss = base_loss + lam * chiral_loss

        self.log('val/loss',        total_loss,  on_epoch=True)
        self.log('val/base_loss',   base_loss,   on_epoch=True)
        self.log('val/chiral_loss', chiral_loss, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer / scheduler — inherit Boltz-2 defaults but allow override
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg = self.config
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=cfg.max_steps,
            eta_min=cfg.min_learning_rate,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
