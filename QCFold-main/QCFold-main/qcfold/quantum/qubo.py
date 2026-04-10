"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation
for fold-switching state assignment.

The fold-switching state assignment problem:
  Given a protein region of n residues that can adopt two distinct
  conformations (Fold A and Fold B), assign each residue to one fold
  state such that the total energy (structural consistency + physics
  constraints) is minimized.

This maps naturally to an Ising model:
  H = sum_i h_i * z_i + sum_{i<j} J_ij * z_i * z_j

Where:
  z_i in {-1, +1} (or equivalently x_i in {0, 1})
  h_i = local energy preference for Fold A vs Fold B at residue i
  J_ij = pairwise coupling: preference for same/different fold states
         based on distance and contact geometry

The cost function includes:
  1. Local torsion angle energy (Ramachandran preference)
  2. Pairwise contact consistency (same-fold neighbors should be compatible)
  3. Steric clash penalty (mixed-fold interfaces)
  4. Boundary penalty (fold-switch boundaries should be at flexible regions)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class QUBOInstance:
    """A QUBO problem instance for fold-state assignment."""
    n_residues: int
    Q: np.ndarray            # (n, n) QUBO matrix
    linear: np.ndarray       # (n,) linear terms (diagonal of Q)
    quadratic: np.ndarray    # (n, n) quadratic terms (off-diagonal)
    offset: float            # Constant energy offset
    residue_indices: np.ndarray  # Original residue indices
    fold_a_torsions: np.ndarray  # (n, 2) phi/psi for Fold A
    fold_b_torsions: np.ndarray  # (n, 2) phi/psi for Fold B

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the QUBO cost for a binary assignment x in {0,1}^n."""
        return float(x @ self.Q @ x + self.offset)

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising form: H = sum h_i z_i + sum J_ij z_i z_j + c.
        
        Using x = (1 + z) / 2 transformation.
        """
        n = self.n_residues
        h = np.zeros(n)
        J = np.zeros((n, n))
        c = self.offset

        for i in range(n):
            h[i] = self.Q[i, i] / 2
            for j in range(n):
                if i != j:
                    h[i] += self.Q[i, j] / 4
                    h[j] += self.Q[i, j] / 4
                    J[i, j] = self.Q[i, j] / 4
            c += self.Q[i, i] / 2

        return h, J, c


def build_fold_switch_qubo(
    fold_a_coords: np.ndarray,     # (n, 3) CA coords for Fold A
    fold_b_coords: np.ndarray,     # (n, 3) CA coords for Fold B
    fold_a_torsions: np.ndarray,   # (n, 2) phi/psi for Fold A
    fold_b_torsions: np.ndarray,   # (n, 2) phi/psi for Fold B
    residue_indices: np.ndarray,
    contact_threshold: float = 8.0,
    clash_threshold: float = 3.0,
    boundary_flexibility: Optional[np.ndarray] = None,  # B-factors
    weights: Optional[Dict[str, float]] = None,
) -> QUBOInstance:
    """Build a QUBO instance for the fold-state assignment problem.
    
    Binary variable x_i: 0 = assign residue i to Fold A, 1 = Fold B.
    
    Cost terms:
      1. Local energy: h_i = E_rama(fold_b_i) - E_rama(fold_a_i)
         Positive h_i means Fold A is locally preferred.
      2. Contact consistency: J_ij rewards same-fold assignment for
         residue pairs that are in contact in BOTH folds, and penalizes
         mixed assignments for close contacts.
      3. Steric clash: Penalizes mixed-fold boundaries where the hybrid
         structure would have atomic overlaps.
      4. Boundary penalty: Encourages fold-switch boundaries at flexible
         (high B-factor) regions.
    """
    if weights is None:
        weights = {
            "local": 1.0,
            "contact": 2.0,
            "clash": 5.0,
            "boundary": 1.0,
        }

    n = len(fold_a_coords)
    Q = np.zeros((n, n))

    # 1. Local energy terms (diagonal)
    for i in range(n):
        phi_a, psi_a = fold_a_torsions[i]
        phi_b, psi_b = fold_b_torsions[i]
        # Ramachandran energy approximation: prefer angles in allowed regions
        e_a = _ramachandran_energy(phi_a, psi_a)
        e_b = _ramachandran_energy(phi_b, psi_b)
        # x_i = 0 → Fold A (cost 0), x_i = 1 → Fold B (cost e_b - e_a)
        Q[i, i] += weights["local"] * (e_b - e_a)

    # 2. Pairwise contact consistency (off-diagonal)
    dist_a = np.sqrt(np.sum((fold_a_coords[:, None] - fold_a_coords[None, :])**2, axis=-1))
    dist_b = np.sqrt(np.sum((fold_b_coords[:, None] - fold_b_coords[None, :])**2, axis=-1))

    for i in range(n):
        for j in range(i + 1, n):
            contact_a = dist_a[i, j] < contact_threshold
            contact_b = dist_b[i, j] < contact_threshold

            if contact_a and contact_b:
                # Both folds have this contact → reward same assignment
                # x_i = x_j → 0; x_i != x_j → penalty
                # Q[i,j] = penalty (added when x_i*x_j = 1 AND both = 1)
                # For "same assignment" reward: Q[i,j] = -w, Q[i,i] += w/2, Q[j,j] += w/2
                w = weights["contact"] / max(dist_a[i, j], 1.0)
                Q[i, j] += w
                Q[j, i] += w
                Q[i, i] -= w
                Q[j, j] -= w
            elif contact_a != contact_b:
                # Contact exists in one fold but not the other → penalize mixing
                w = weights["contact"] * 0.5
                Q[i, j] += w
                Q[j, i] += w

    # 3. Steric clash penalty for mixed-fold boundaries
    for i in range(n - 1):
        j = i + 1
        # If consecutive residues are in different folds, check for clash
        # at the boundary. This is approximated by the distance mismatch.
        mixed_dist = np.linalg.norm(fold_a_coords[i] - fold_b_coords[j])
        if mixed_dist < clash_threshold:
            penalty = weights["clash"] * (clash_threshold - mixed_dist)
            # Penalize x_i != x_j: add Q[i,j] = penalty
            Q[i, j] += penalty
            Q[j, i] += penalty
            Q[i, i] -= penalty
            Q[j, j] -= penalty

    # 4. Boundary flexibility penalty
    if boundary_flexibility is not None:
        # Higher B-factor → lower penalty for fold switching at this residue
        max_bf = np.max(boundary_flexibility) + 1e-8
        normalized_bf = boundary_flexibility / max_bf
        for i in range(n - 1):
            # Reduce boundary penalty at flexible residues
            flexibility_discount = (normalized_bf[i] + normalized_bf[i+1]) / 2
            Q[i, i+1] *= (1.0 - weights["boundary"] * flexibility_discount)
            Q[i+1, i] *= (1.0 - weights["boundary"] * flexibility_discount)

    # Make Q symmetric
    Q = (Q + Q.T) / 2

    return QUBOInstance(
        n_residues=n,
        Q=Q,
        linear=np.diag(Q),
        quadratic=Q - np.diag(np.diag(Q)),
        offset=0.0,
        residue_indices=residue_indices,
        fold_a_torsions=fold_a_torsions,
        fold_b_torsions=fold_b_torsions,
    )


def _ramachandran_energy(phi: float, psi: float) -> float:
    """Approximate Ramachandran energy for a (phi, psi) pair.
    
    Uses a simplified Gaussian mixture model with three basins:
      - alpha helix: (phi=-60°, psi=-47°)
      - beta sheet:  (phi=-120°, psi=130°)
      - left-handed: (phi=60°, psi=40°)
    
    Returns energy (lower = more favorable). Range approximately [0, 5].
    """
    # Convert to degrees for interpretability
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)

    # Gaussian basins
    basins = [
        (-60, -47, 25, 25, 0.0),    # alpha helix (most favorable)
        (-120, 130, 30, 30, 0.5),    # beta sheet
        (60, 40, 20, 20, 1.5),       # left-handed helix
        (-80, 150, 25, 25, 0.8),     # polyproline II
    ]

    min_energy = 5.0
    for phi0, psi0, sig_phi, sig_psi, base_e in basins:
        dphi = _angle_diff_deg(phi_deg, phi0)
        dpsi = _angle_diff_deg(psi_deg, psi0)
        e = base_e + (dphi**2 / (2 * sig_phi**2)) + (dpsi**2 / (2 * sig_psi**2))
        min_energy = min(min_energy, e)

    return min_energy


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angular difference in degrees, wrapped to [-180, 180]."""
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d
