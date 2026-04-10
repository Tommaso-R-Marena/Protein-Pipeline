"""Configuration dataclass for ChiralBoltz fine-tuning."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChiralBoltzConfig:
    # --- Chiral loss ---
    lambda_chiral: float = 1.0           # Final weight of chiral volume loss
    chiral_loss_warmup_steps: int = 5000 # Steps to linearly warm up lambda_chiral
    chiral_loss_mode: str = "combined"   # "sign" | "mse" | "combined"
    chiral_sign_margin: float = 0.1      # Hinge margin for sign loss
    chiral_mse_weight: float = 0.1       # MSE term weight in "combined" mode

    # --- Mirror augmentation ---
    p_mirror: float = 0.5                # Probability of applying mirror augmentation per sample
    mirror_seed: int = 42

    # --- Optimizer ---
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_steps: int = 50_000

    # --- Boltz-2 checkpoint ---
    boltz2_checkpoint: Optional[str] = None   # Path to Boltz-2 pre-trained weights

    # --- Data ---
    target_dir: str = "data/structures"
    msa_dir: str    = "data/msas"
    max_tokens: int = 512
    max_atoms: int  = 4608

    # --- Training ---
    batch_size: int  = 4
    num_workers: int = 4
    devices: int     = 1
    precision: str   = "bf16-mixed"
    output_dir: str  = "outputs/chiralboltz"
