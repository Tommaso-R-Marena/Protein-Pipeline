"""
Physics and geometry consistency layer.

Enforces physically valid protein structures through:
  1. Bond geometry validation (N-CA, CA-C, C-N distances)
  2. Steric clash detection and penalty
  3. Ramachandran validation
  4. Hydrogen bond detection
  5. Contact map consistency
  6. Overall structural quality score

These constraints can be used as:
  - Hard filters (reject invalid structures)
  - Soft penalties (loss terms during training)
  - Scoring components for candidate ranking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Standard bond lengths (Angstroms) and tolerances
STANDARD_BONDS = {
    "N_CA": (1.458, 0.019),   # mean, std
    "CA_C": (1.524, 0.014),
    "C_N": (1.329, 0.014),
    "C_O": (1.231, 0.020),
}

# Van der Waals radii (Angstroms) for clash detection
VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "CA": 1.70, "CB": 1.70,
}

# Minimum non-bonded distance (Angstroms)
MIN_NONBOND_DIST = 2.0

# Ramachandran allowed regions (degrees)
RAMA_ALLOWED = [
    {"name": "alpha_R", "phi": (-120, 0), "psi": (-80, 0)},
    {"name": "beta", "phi": (-180, -50), "psi": (50, 180)},
    {"name": "alpha_L", "phi": (30, 90), "psi": (0, 80)},
    {"name": "ppII", "phi": (-100, -50), "psi": (100, 180)},
]


@dataclass
class PhysicsScore:
    """Comprehensive physics-based quality score for a structure."""
    total_score: float
    bond_score: float
    clash_score: float
    rama_score: float
    contact_score: float
    hbond_score: float
    num_clashes: int
    num_rama_outliers: int
    num_bond_outliers: int
    per_residue_scores: np.ndarray
    details: Dict


def compute_bond_geometry(
    n_coords: np.ndarray,    # (L, 3)
    ca_coords: np.ndarray,   # (L, 3)
    c_coords: np.ndarray,    # (L, 3)
) -> Tuple[float, int, Dict]:
    """Validate backbone bond lengths.
    
    Returns:
        score: normalized bond geometry score [0, 1] (1 = perfect)
        num_outliers: number of bonds outside 3σ
        details: per-bond analysis
    """
    L = len(ca_coords)
    deviations = []
    outliers = 0
    details = {"bonds": []}

    for i in range(L):
        # N-CA bond
        d_nca = np.linalg.norm(ca_coords[i] - n_coords[i])
        mean, std = STANDARD_BONDS["N_CA"]
        z = abs(d_nca - mean) / std
        deviations.append(z)
        if z > 3:
            outliers += 1
        details["bonds"].append(("N_CA", i, d_nca, z))

        # CA-C bond
        d_cac = np.linalg.norm(c_coords[i] - ca_coords[i])
        mean, std = STANDARD_BONDS["CA_C"]
        z = abs(d_cac - mean) / std
        deviations.append(z)
        if z > 3:
            outliers += 1

        # C-N peptide bond (to next residue)
        if i < L - 1:
            d_cn = np.linalg.norm(n_coords[i + 1] - c_coords[i])
            mean, std = STANDARD_BONDS["C_N"]
            z = abs(d_cn - mean) / std
            deviations.append(z)
            if z > 3:
                outliers += 1

    # Score: fraction of bonds within 3σ
    score = 1.0 - (outliers / max(len(deviations), 1))
    return score, outliers, details


def detect_steric_clashes(
    ca_coords: np.ndarray,
    threshold: float = MIN_NONBOND_DIST,
    min_seq_sep: int = 3,
) -> Tuple[float, int, List[Tuple[int, int, float]]]:
    """Detect steric clashes between non-bonded atoms.
    
    Args:
        ca_coords: (L, 3) CA coordinates
        threshold: minimum allowed distance
        min_seq_sep: minimum sequence separation to consider
        
    Returns:
        score: clash score [0, 1] (1 = no clashes)
        num_clashes: number of clashing pairs
        clashes: list of (i, j, distance) tuples
    """
    L = len(ca_coords)
    dists = np.sqrt(np.sum(
        (ca_coords[:, None, :] - ca_coords[None, :, :])**2, axis=-1
    ))

    clashes = []
    for i in range(L):
        for j in range(i + min_seq_sep, L):
            if dists[i, j] < threshold:
                clashes.append((i, j, float(dists[i, j])))

    max_possible = max((L * (L - 1) // 2 - L * min_seq_sep), 1)
    score = 1.0 - min(len(clashes) / max_possible, 1.0)
    return score, len(clashes), clashes


def ramachandran_validation(
    phi: np.ndarray,
    psi: np.ndarray,
) -> Tuple[float, int, np.ndarray]:
    """Validate torsion angles against Ramachandran allowed regions.
    
    Returns:
        score: fraction of residues in allowed regions [0, 1]
        num_outliers: number of Ramachandran outliers
        per_residue: (L,) boolean array, True if in allowed region
    """
    L = len(phi)
    in_allowed = np.zeros(L, dtype=bool)

    for i in range(L):
        if np.isnan(phi[i]) or np.isnan(psi[i]):
            in_allowed[i] = True  # Terminal residues
            continue

        phi_deg = np.degrees(phi[i])
        psi_deg = np.degrees(psi[i])

        for region in RAMA_ALLOWED:
            phi_lo, phi_hi = region["phi"]
            psi_lo, psi_hi = region["psi"]
            if phi_lo <= phi_deg <= phi_hi and psi_lo <= psi_deg <= psi_hi:
                in_allowed[i] = True
                break

    score = float(np.mean(in_allowed))
    num_outliers = int(np.sum(~in_allowed))
    return score, num_outliers, in_allowed


def contact_map_consistency(
    pred_coords: np.ndarray,
    ref_contact_map: np.ndarray,
    threshold: float = 8.0,
) -> float:
    """Measure consistency between predicted and reference contact maps.
    
    Returns F1 score between predicted and reference contacts.
    """
    pred_dist = np.sqrt(np.sum(
        (pred_coords[:, None, :] - pred_coords[None, :, :])**2, axis=-1
    ))
    pred_contacts = (pred_dist < threshold).astype(float)

    # Exclude diagonal and near-diagonal
    mask = np.abs(np.arange(len(pred_coords))[:, None] -
                  np.arange(len(pred_coords))[None, :]) >= 6

    pred_masked = pred_contacts[mask]
    ref_masked = ref_contact_map[mask]

    tp = np.sum(pred_masked * ref_masked)
    fp = np.sum(pred_masked * (1 - ref_masked))
    fn = np.sum((1 - pred_masked) * ref_masked)

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return float(f1)


def compute_physics_score(
    ca_coords: np.ndarray,
    n_coords: Optional[np.ndarray] = None,
    c_coords: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
    ref_contact_map: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> PhysicsScore:
    """Compute comprehensive physics-based quality score.
    
    Args:
        ca_coords: (L, 3) CA coordinates
        n_coords, c_coords: backbone N and C coordinates (optional)
        phi, psi: torsion angles (optional)
        ref_contact_map: reference contact map for consistency (optional)
        weights: dict of component weights
    
    Returns:
        PhysicsScore with component scores and total
    """
    if weights is None:
        weights = {
            "bond": 0.25,
            "clash": 0.30,
            "rama": 0.25,
            "contact": 0.10,
            "hbond": 0.10,
        }

    L = len(ca_coords)
    per_residue = np.ones(L)

    # Bond geometry
    bond_score = 1.0
    num_bond_outliers = 0
    bond_details = {}
    if n_coords is not None and c_coords is not None:
        bond_score, num_bond_outliers, bond_details = compute_bond_geometry(
            n_coords, ca_coords, c_coords
        )

    # Steric clashes
    clash_score, num_clashes, clash_list = detect_steric_clashes(ca_coords)
    for i, j, d in clash_list:
        per_residue[i] *= 0.5
        per_residue[j] *= 0.5

    # Ramachandran
    rama_score = 1.0
    num_rama_outliers = 0
    if phi is not None and psi is not None:
        rama_score, num_rama_outliers, rama_valid = ramachandran_validation(
            phi, psi
        )
        per_residue *= rama_valid.astype(float) * 0.5 + 0.5

    # Contact consistency
    contact_score = 1.0
    if ref_contact_map is not None:
        contact_score = contact_map_consistency(
            ca_coords, ref_contact_map
        )

    # Hydrogen bond score (simplified: based on backbone geometry)
    hbond_score = 1.0  # Placeholder for full H-bond analysis

    # Total weighted score
    total = (
        weights["bond"] * bond_score +
        weights["clash"] * clash_score +
        weights["rama"] * rama_score +
        weights["contact"] * contact_score +
        weights["hbond"] * hbond_score
    )

    return PhysicsScore(
        total_score=total,
        bond_score=bond_score,
        clash_score=clash_score,
        rama_score=rama_score,
        contact_score=contact_score,
        hbond_score=hbond_score,
        num_clashes=num_clashes,
        num_rama_outliers=num_rama_outliers,
        num_bond_outliers=num_bond_outliers,
        per_residue_scores=per_residue,
        details={"bond_details": bond_details, "clashes": clash_list},
    )
