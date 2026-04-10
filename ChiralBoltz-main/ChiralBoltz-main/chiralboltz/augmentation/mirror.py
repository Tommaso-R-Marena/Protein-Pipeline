"""
Mirror-image augmentation for protein structures.

The key insight: reflecting a coordinate system through x -> -x inverts ALL chiral centers.
An L-protein structure becomes a D-protein structure. Since Boltz-2 processes atom coordinates
as Float[B, N, 3], this is a single sign negation — but requires careful handling of:
  1. Coordinate arrays in the Boltz-2 StructureV2 format (coords, ensemble fields)
  2. Bond chirality flags in RDKit mol objects (for ligands)
  3. The `chirality` field in the atoms structured array
  4. MSA features (unaffected — sequence is sequence-invariant to reflection)
"""

import numpy as np
import torch
from torch import Tensor


REFLECTION_MATRIX = torch.tensor(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 1.0, 0.0],
     [ 0.0, 0.0, 1.0]],
    dtype=torch.float32,
)


def reflect_coords_numpy(coords: np.ndarray) -> np.ndarray:
    """
    Reflect coordinates through the y-z plane (negate x).

    Parameters
    ----------
    coords : np.ndarray
        Shape (..., 3). Any leading batch/ensemble dimensions supported.

    Returns
    -------
    np.ndarray
        Reflected coordinates, same shape as input.
    """
    out = coords.copy()
    out[..., 0] = -out[..., 0]
    return out


def reflect_coords_tensor(coords: Tensor) -> Tensor:
    """
    Reflect coordinates through y-z plane. Differentiable.

    Parameters
    ----------
    coords : Tensor
        Shape (..., 3).

    Returns
    -------
    Tensor
        Reflected coordinates with gradients preserved.
    """
    # Multiply only x-channel by -1, preserving grad_fn
    sign = coords.new_ones(coords.shape)
    sign[..., 0] = -1.0
    return coords * sign


def flip_chirality_flags(chirality_array: np.ndarray) -> np.ndarray:
    """
    Invert chirality flags in a Boltz-2 atoms structured array.

    In the Boltz-2 atoms dtype, the 'chirality' field uses integer codes:
      0 = no chirality / achiral
      1 = CW  (clockwise,  R in CIP)
      2 = CCW (counter-clockwise, S in CIP)
    After a mirror reflection, CW <-> CCW, so 1 <-> 2.

    Parameters
    ----------
    chirality_array : np.ndarray
        1-D structured array from StructureV2.atoms, must contain 'chirality' field.

    Returns
    -------
    np.ndarray
        Copy with chirality flags inverted.
    """
    out = chirality_array.copy()
    mask_cw  = out['chirality'] == 1
    mask_ccw = out['chirality'] == 2
    out['chirality'][mask_cw]  = 2
    out['chirality'][mask_ccw] = 1
    return out


def mirror_structure_v2(structure) -> object:
    """
    Produce a mirror image of a Boltz-2 StructureV2 object.

    - Reflects coords and ensemble arrays.
    - Inverts chirality flags in atoms.
    - All other fields (bonds, residues, chains, interfaces, mask, pocket) are unchanged.

    Parameters
    ----------
    structure : StructureV2
        From boltz.data.types.

    Returns
    -------
    StructureV2
        New structure with all coordinates reflected.
    """
    from boltz.data.types import StructureV2

    mirrored_coords   = reflect_coords_numpy(structure.coords)
    mirrored_ensemble = reflect_coords_numpy(structure.ensemble)
    mirrored_atoms    = flip_chirality_flags(structure.atoms)

    return StructureV2(
        atoms=mirrored_atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        interfaces=structure.interfaces,
        mask=structure.mask,
        coords=mirrored_coords,
        ensemble=mirrored_ensemble,
        pocket=structure.pocket,
    )
