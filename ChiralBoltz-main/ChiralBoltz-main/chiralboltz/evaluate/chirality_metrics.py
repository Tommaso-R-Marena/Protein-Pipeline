"""
ChiralFold-style chirality metrics for evaluating structure predictions.

Implements the signed chiral volume approach from:
  ChiralFold (Marena 2026): https://github.com/Tommaso-R-Marena/ChiralFold
  Ishitani & Moriwaki (ACS Omega, 2025)

Key metrics:
  - per_residue_chirality_correct: fraction of residues with correct chiral sign
  - chirality_violation_rate: complement of the above (matches Childs et al. reporting)
  - mean_chiral_volume_magnitude: sanity check for planar/flat centers
"""

import torch
from torch import Tensor
from chiralboltz.loss.chiral_volume import signed_chiral_volume


def chirality_violation_rate(
    pred_coords: Tensor,     # Float[N_atoms, 3]
    true_coords: Tensor,     # Float[N_atoms, 3]
    chiral_centers: Tensor,  # Long[N_chiral, 4]  — [center, j, k, l]
    eps: float = 1e-3,
) -> dict:
    """
    Compute per-center chirality metrics.

    Returns
    -------
    dict with keys:
      'violation_rate'       : float in [0, 1] — fraction of centers violated
      'correct_rate'         : float in [0, 1]
      'planar_rate'          : float in [0, 1] — |V| < eps (undetermined chirality)
      'n_centers'            : int
      'mean_volume_pred'     : float
      'mean_volume_true'     : float
    """
    if chiral_centers.shape[0] == 0:
        return {'violation_rate': 0.0, 'correct_rate': 1.0, 'planar_rate': 0.0,
                'n_centers': 0, 'mean_volume_pred': 0.0, 'mean_volume_true': 0.0}

    with torch.no_grad():
        # Add batch dimension for the loss function signature
        def _vol(coords, centers):
            c   = coords[centers[:, 0]]
            nb1 = coords[centers[:, 1]]
            nb2 = coords[centers[:, 2]]
            nb3 = coords[centers[:, 3]]
            return signed_chiral_volume(c, nb1, nb2, nb3)

        V_pred = _vol(pred_coords, chiral_centers)
        V_true = _vol(true_coords, chiral_centers)

        planar  = V_pred.abs() < eps
        correct = (torch.sign(V_pred) == torch.sign(V_true)) & ~planar
        violated = ~correct

    n = len(chiral_centers)
    return {
        'violation_rate':    violated.float().mean().item(),
        'correct_rate':      correct.float().mean().item(),
        'planar_rate':       planar.float().mean().item(),
        'n_centers':         n,
        'mean_volume_pred':  V_pred.abs().mean().item(),
        'mean_volume_true':  V_true.abs().mean().item(),
    }
