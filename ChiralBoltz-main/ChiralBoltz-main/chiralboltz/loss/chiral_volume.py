"""
Differentiable chiral volume loss for heterochiral structure prediction.

Notation follows ChiralFold (Marena 2026) and Ishitani & Moriwaki (ACS Omega, 2025).

Reference: ChiralFold signed chiral volume validator:
  https://github.com/Tommaso-R-Marena/ChiralFold
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def signed_chiral_volume(
    r_center: Tensor,    # Float[..., 3]  — α-carbon or chiral atom
    r_j: Tensor,         # Float[..., 3]  — neighbor 1
    r_k: Tensor,         # Float[..., 3]  — neighbor 2
    r_l: Tensor,         # Float[..., 3]  — neighbor 3
) -> Tensor:             # Float[...]     — signed volume
    """
    Compute the signed scalar triple product (chiral volume).

    V = (r_c - r_j) · ((r_c - r_k) × (r_c - r_l))

    This is fully differentiable w.r.t. all coordinate inputs.

    Parameters
    ----------
    r_center, r_j, r_k, r_l : Tensor
        Coordinates of the chiral center and its three substituents.
        Arbitrary leading batch dimensions supported.

    Returns
    -------
    Tensor
        Signed scalar triple product per chiral center.
    """
    v1 = r_center - r_j
    v2 = r_center - r_k
    v3 = r_center - r_l
    return torch.sum(v1 * torch.linalg.cross(v2, v3), dim=-1)


def chiral_volume_loss(
    pred_coords: Tensor,      # Float[B, N_atoms, 3]
    true_coords: Tensor,      # Float[B, N_atoms, 3]
    chiral_centers: Tensor,   # Long[B, N_chiral, 4]  — indices: [center, j, k, l]
    chiral_mask: Tensor,      # Bool[B, N_chiral]     — valid centers
    mode: str = "sign",       # "sign" | "mse" | "combined"
    sign_margin: float = 0.1,
    mse_weight: float = 0.1,
) -> Tensor:
    """
    Differentiable chiral volume loss.

    Penalizes:
      - "sign":     hinge loss on volume sign disagreement (soft, differentiable)
      - "mse":      MSE between predicted and reference volumes
      - "combined": weighted sum of sign + mse

    The sign loss uses a hinge formulation:
      L_sign = max(0, margin - V_pred * sign(V_true))
    so that the loss is zero when V_pred has the correct sign with margin,
    and increases as V_pred approaches zero or flips sign.

    Parameters
    ----------
    pred_coords : Tensor
        Predicted atom coordinates from the diffusion model.
    true_coords : Tensor
        Ground-truth atom coordinates.
    chiral_centers : Tensor
        Indices of chiral center atoms and three substituents.
    chiral_mask : Tensor
        Boolean mask for valid (non-padding) chiral centers.
    mode : str
        Loss computation mode.
    sign_margin : float
        Margin for the hinge sign loss.
    mse_weight : float
        Weight for MSE term in "combined" mode.

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    if chiral_mask.sum() == 0:
        return pred_coords.new_zeros(())

    idx_c = chiral_centers[..., 0]   # center atom index
    idx_j = chiral_centers[..., 1]
    idx_k = chiral_centers[..., 2]
    idx_l = chiral_centers[..., 3]

    def gather_coords(coords: Tensor, idx: Tensor) -> Tensor:
        # idx: Long[B, N_chiral]
        # coords: Float[B, N_atoms, 3]
        B, N_chiral = idx.shape
        idx_exp = idx.unsqueeze(-1).expand(B, N_chiral, 3)
        return coords.gather(1, idx_exp)   # Float[B, N_chiral, 3]

    # Predicted volumes
    V_pred = signed_chiral_volume(
        gather_coords(pred_coords, idx_c),
        gather_coords(pred_coords, idx_j),
        gather_coords(pred_coords, idx_k),
        gather_coords(pred_coords, idx_l),
    )  # Float[B, N_chiral]

    # Reference volumes from true coordinates
    with torch.no_grad():
        V_true = signed_chiral_volume(
            gather_coords(true_coords, idx_c),
            gather_coords(true_coords, idx_j),
            gather_coords(true_coords, idx_k),
            gather_coords(true_coords, idx_l),
        )  # Float[B, N_chiral]

    # Mask padding
    V_pred = V_pred * chiral_mask
    V_true = V_true * chiral_mask

    if mode == "sign":
        # Hinge loss: penalise when predicted sign disagrees with reference
        sign_true = torch.sign(V_true)
        loss = F.relu(sign_margin - V_pred * sign_true)
        return loss[chiral_mask].mean()

    elif mode == "mse":
        loss = (V_pred - V_true) ** 2
        return loss[chiral_mask].mean()

    elif mode == "combined":
        sign_true = torch.sign(V_true)
        sign_loss = F.relu(sign_margin - V_pred * sign_true)[chiral_mask].mean()
        mse_loss  = ((V_pred - V_true) ** 2)[chiral_mask].mean()
        return sign_loss + mse_weight * mse_loss

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'sign', 'mse', or 'combined'.")


def extract_chiral_centers_from_batch(featurized_batch: dict) -> tuple[Tensor, Tensor]:
    """
    Extract chiral center index tensors from a Boltz-2 featurized batch.

    Boltz-2 featurized batches include an 'atom_chirality' field in features
    that records the CIP chirality of each atom (0=none, 1=CW, 2=CCW).
    We identify atoms with non-zero chirality and retrieve their three
    bonded heavy-atom neighbors from the bond graph.

    Parameters
    ----------
    featurized_batch : dict
        Standard Boltz-2 featurized batch with keys:
          'atom_chirality': Long[B, N_atoms]
          'bonds': Long[B, N_bonds, 2]    (atom index pairs)
          'atom_mask': Bool[B, N_atoms]

    Returns
    -------
    chiral_centers : Long[B, N_chiral, 4]
        Atom indices [center, j, k, l] for each chiral center.
    chiral_mask : Bool[B, N_chiral]
        Valid center mask (False for padding).

    Notes
    -----
    This function builds a neighbor list from the bond tensor at runtime.
    For training efficiency, consider pre-computing and caching this in the
    dataset pipeline.
    """
    atom_chirality = featurized_batch['atom_chirality']  # Long[B, N]
    bonds          = featurized_batch['bonds']           # Long[B, N_bonds, 2]
    atom_mask      = featurized_batch['atom_mask']       # Bool[B, N]

    B, N = atom_chirality.shape
    device = atom_chirality.device

    all_centers  = []
    max_n_chiral = 0

    for b in range(B):
        # Find chiral atoms for this batch element
        chiral_atom_indices = ((atom_chirality[b] != 0) & atom_mask[b]).nonzero(as_tuple=False).squeeze(-1)

        # Build adjacency: for each atom, list of bonded atoms
        neighbors: dict[int, list[int]] = {}
        valid_bonds = bonds[b]  # Long[N_bonds, 2]
        for bond_idx in range(valid_bonds.shape[0]):
            a, c = valid_bonds[bond_idx, 0].item(), valid_bonds[bond_idx, 1].item()
            if a == -1 or c == -1:
                continue
            neighbors.setdefault(a, []).append(c)
            neighbors.setdefault(c, []).append(a)

        centers_b = []
        for ci in chiral_atom_indices.tolist():
            nbrs = neighbors.get(ci, [])
            if len(nbrs) >= 3:
                nb1, nb2, nb3 = nbrs[:3]
                centers_b.append([ci, nb1, nb2, nb3])

        all_centers.append(centers_b)
        max_n_chiral = max(max_n_chiral, len(centers_b))

    # Pad to max_n_chiral
    chiral_centers = torch.zeros(B, max(max_n_chiral, 1), 4, dtype=torch.long, device=device)
    chiral_mask    = torch.zeros(B, max(max_n_chiral, 1), dtype=torch.bool, device=device)

    for b, centers_b in enumerate(all_centers):
        for i, quad in enumerate(centers_b):
            chiral_centers[b, i] = torch.tensor(quad, dtype=torch.long, device=device)
            chiral_mask[b, i]    = True

    return chiral_centers, chiral_mask
