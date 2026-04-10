"""
Ensemble prediction and uncertainty-aware ranking.

Generates diverse structural conformations and provides:
  1. Multiple candidate structures with diversity enforcement
  2. Per-structure confidence scores
  3. Per-residue uncertainty estimates
  4. Uncertainty-aware ranking that balances quality and confidence
  5. Calibrated predictions for fold-switching proteins
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .physics_layer import compute_physics_score, PhysicsScore


@dataclass
class EnsemblePrediction:
    """A set of predicted conformations with uncertainty estimates."""
    structures: List[np.ndarray]           # List of (L, 3) coordinate arrays
    confidences: np.ndarray                 # (K,) per-structure confidence
    per_residue_uncertainty: np.ndarray     # (L,) per-residue uncertainty
    physics_scores: List[PhysicsScore]      # Physics score per structure
    rankings: np.ndarray                    # (K,) ranking indices (best first)
    diversity_score: float                  # Average pairwise diversity
    torsion_sets: Optional[List[np.ndarray]] = None  # Optional torsion angles


class EnsembleGenerator:
    """Generate and rank diverse structural ensembles."""

    def __init__(
        self,
        num_conformations: int = 8,
        temperature: float = 1.0,
        diversity_weight: float = 0.3,
        physics_weight: float = 0.4,
        learned_weight: float = 0.4,
        uncertainty_weight: float = 0.2,
    ):
        self.num_conformations = num_conformations
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.physics_weight = physics_weight
        self.learned_weight = learned_weight
        self.uncertainty_weight = uncertainty_weight

    def generate_ensemble(
        self,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
        assignments: List[np.ndarray],  # List of fold-state assignments
        fold_a_torsions: Optional[np.ndarray] = None,
        fold_b_torsions: Optional[np.ndarray] = None,
        n_coords: Optional[np.ndarray] = None,
        c_coords: Optional[np.ndarray] = None,
        learned_scores: Optional[np.ndarray] = None,
    ) -> EnsemblePrediction:
        """Generate an ensemble of diverse conformations.
        
        Args:
            fold_a_coords: (L, 3) Fold A reference
            fold_b_coords: (L, 3) Fold B reference
            assignments: list of (L,) binary fold-state assignments
            fold_a_torsions, fold_b_torsions: optional torsion angles
            n_coords, c_coords: backbone atoms for physics scoring
            learned_scores: optional pre-computed learned quality scores
        """
        L = len(fold_a_coords)
        K = len(assignments)

        # Decode each assignment to coordinates
        structures = []
        torsion_sets = []
        for assignment in assignments:
            coords = np.where(
                assignment[:, None] == 0,
                fold_a_coords,
                fold_b_coords,
            )
            structures.append(coords)

            if fold_a_torsions is not None and fold_b_torsions is not None:
                torsions = np.where(
                    assignment[:, None] == 0,
                    fold_a_torsions,
                    fold_b_torsions,
                )
                torsion_sets.append(torsions)

        # Compute physics scores
        physics_scores = []
        for i, coords in enumerate(structures):
            ps = compute_physics_score(
                ca_coords=coords,
                n_coords=n_coords,
                c_coords=c_coords,
                phi=torsion_sets[i][:, 0] if torsion_sets else None,
                psi=torsion_sets[i][:, 1] if torsion_sets else None,
            )
            physics_scores.append(ps)

        # Compute per-structure confidence
        phys_array = np.array([ps.total_score for ps in physics_scores])
        if learned_scores is not None:
            combined_scores = (
                self.physics_weight * phys_array +
                self.learned_weight * learned_scores[:K]
            )
        else:
            combined_scores = phys_array

        # Softmax for confidence
        confidences = _softmax(combined_scores / self.temperature)

        # Per-residue uncertainty
        per_residue_uncertainty = self._compute_residue_uncertainty(
            structures, assignments
        )

        # Diversity-aware ranking
        rankings = self._diversity_rank(
            structures, combined_scores, assignments
        )

        # Ensemble diversity
        diversity = self._compute_diversity(structures)

        return EnsemblePrediction(
            structures=[structures[i] for i in rankings],
            confidences=confidences[rankings],
            per_residue_uncertainty=per_residue_uncertainty,
            physics_scores=[physics_scores[i] for i in rankings],
            rankings=rankings,
            diversity_score=diversity,
            torsion_sets=[torsion_sets[i] for i in rankings] if torsion_sets else None,
        )

    def _compute_residue_uncertainty(
        self,
        structures: List[np.ndarray],
        assignments: List[np.ndarray],
    ) -> np.ndarray:
        """Compute per-residue uncertainty from ensemble variance.
        
        Uncertainty is high where:
          - Structures disagree on coordinates (high variance)
          - Assignments disagree on fold state (fold-switching residues)
        """
        if len(structures) < 2:
            return np.zeros(len(structures[0]))

        coords_stack = np.stack(structures)  # (K, L, 3)
        assign_stack = np.stack(assignments)  # (K, L)

        # Coordinate variance
        coord_var = np.mean(np.var(coords_stack, axis=0), axis=1)  # (L,)

        # Assignment entropy (binary entropy)
        p_fold_b = np.mean(assign_stack, axis=0)  # (L,)
        entropy = -p_fold_b * np.log2(p_fold_b + 1e-10) - \
                  (1 - p_fold_b) * np.log2(1 - p_fold_b + 1e-10)

        # Combine (normalized)
        max_var = np.max(coord_var) + 1e-10
        uncertainty = 0.5 * (coord_var / max_var) + 0.5 * entropy

        return uncertainty

    def _diversity_rank(
        self,
        structures: List[np.ndarray],
        scores: np.ndarray,
        assignments: List[np.ndarray],
    ) -> np.ndarray:
        """Rank structures balancing quality and diversity.
        
        Uses a greedy selection: start with the best-scoring structure,
        then iteratively add the structure that maximizes
        score + diversity_weight * min_distance_to_selected.
        """
        K = len(structures)
        if K <= 1:
            return np.arange(K)

        selected = []
        remaining = list(range(K))

        # Start with the best-scoring structure
        best_idx = int(np.argmax(scores))
        selected.append(best_idx)
        remaining.remove(best_idx)

        while remaining and len(selected) < self.num_conformations:
            best_score = -float("inf")
            best_candidate = remaining[0]

            for idx in remaining:
                # Quality score
                quality = scores[idx]

                # Minimum distance to any selected structure
                min_dist = float("inf")
                for sel_idx in selected:
                    dist = np.sqrt(np.mean(
                        np.sum((structures[idx] - structures[sel_idx]) ** 2, axis=1)
                    ))
                    min_dist = min(min_dist, dist)

                # Combined score
                combined = quality + self.diversity_weight * min_dist
                if combined > best_score:
                    best_score = combined
                    best_candidate = idx

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return np.array(selected)

    def _compute_diversity(self, structures: List[np.ndarray]) -> float:
        """Average pairwise RMSD within the ensemble."""
        if len(structures) < 2:
            return 0.0

        rmsds = []
        for i in range(len(structures)):
            for j in range(i + 1, len(structures)):
                diff = structures[i] - structures[j]
                rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                rmsds.append(rmsd)

        return float(np.mean(rmsds))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
