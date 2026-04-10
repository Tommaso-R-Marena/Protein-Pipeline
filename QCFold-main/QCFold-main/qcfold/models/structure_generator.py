"""
Multi-conformation structure generator.

Generates diverse backbone conformations using:
  1. Template-based generation from PDB homologs
  2. Torsion angle perturbation (for ensemble diversity)
  3. Fragment-based assembly
  4. Noise injection for diffusion-like diversity

For fold-switching proteins, the key insight is that we need to generate
structures that span BOTH conformational states. This requires:
  - MSA-depth-aware generation (shallow MSA → more diverse outputs)
  - Torsion space sampling across multiple Ramachandran basins
  - Explicit diversity enforcement
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GeneratedStructure:
    """A generated backbone structure with metadata."""
    ca_coords: np.ndarray         # (L, 3)
    torsions: np.ndarray          # (L, 2) phi/psi
    source: str                   # "template", "perturbed", "fragment", "noise"
    score: float                  # Initial quality score
    template_pdb: Optional[str]   # Source template PDB ID if applicable


class MultiConformationGenerator:
    """Generate diverse backbone conformations for fold-switching prediction.
    
    Strategy:
    1. If template structures are available, use them as starting points
    2. Generate perturbations by sampling torsion angles
    3. Add noise-injected variants for diversity
    4. Enforce minimum pairwise diversity
    """

    def __init__(
        self,
        num_candidates: int = 32,
        perturbation_scale: float = 0.5,
        noise_scale: float = 1.0,
        torsion_bins: int = 8,
        min_diversity_rmsd: float = 2.0,
        seed: int = 42,
    ):
        self.num_candidates = num_candidates
        self.perturbation_scale = perturbation_scale
        self.noise_scale = noise_scale
        self.torsion_bins = torsion_bins
        self.min_diversity_rmsd = min_diversity_rmsd
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        sequence: str,
        template_coords: Optional[np.ndarray] = None,
        template_torsions: Optional[np.ndarray] = None,
        fold_a_coords: Optional[np.ndarray] = None,
        fold_b_coords: Optional[np.ndarray] = None,
        fold_a_torsions: Optional[np.ndarray] = None,
        fold_b_torsions: Optional[np.ndarray] = None,
    ) -> List[GeneratedStructure]:
        """Generate diverse backbone conformations.
        
        If both fold coordinates are provided (oracle setting for evaluation),
        generates interpolations and perturbations spanning both states.
        
        If only a template is provided, generates perturbations around it.
        If nothing is provided, generates from torsion angle sampling.
        """
        L = len(sequence)
        structures = []

        # Strategy 1: Use known fold structures if available
        if fold_a_coords is not None and fold_b_coords is not None:
            structures.extend(self._generate_from_dual_folds(
                fold_a_coords, fold_b_coords,
                fold_a_torsions, fold_b_torsions,
            ))

        # Strategy 2: Template perturbation
        if template_coords is not None:
            structures.extend(self._perturb_template(
                template_coords, template_torsions,
            ))

        # Strategy 3: Torsion angle sampling
        if len(structures) < self.num_candidates:
            remaining = self.num_candidates - len(structures)
            structures.extend(self._sample_torsion_space(
                L, remaining,
                reference_torsions=template_torsions or
                    (fold_a_torsions if fold_a_torsions is not None else None),
            ))

        # Enforce diversity
        structures = self._enforce_diversity(structures)

        return structures[:self.num_candidates]

    def _generate_from_dual_folds(
        self,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
        fold_a_torsions: Optional[np.ndarray],
        fold_b_torsions: Optional[np.ndarray],
    ) -> List[GeneratedStructure]:
        """Generate structures spanning both fold states."""
        L = len(fold_a_coords)
        structures = []

        # Pure Fold A
        structures.append(GeneratedStructure(
            ca_coords=fold_a_coords.copy(),
            torsions=fold_a_torsions.copy() if fold_a_torsions is not None
                else np.zeros((L, 2)),
            source="template", score=1.0, template_pdb="fold_a",
        ))

        # Pure Fold B
        structures.append(GeneratedStructure(
            ca_coords=fold_b_coords.copy(),
            torsions=fold_b_torsions.copy() if fold_b_torsions is not None
                else np.zeros((L, 2)),
            source="template", score=1.0, template_pdb="fold_b",
        ))

        # Interpolations
        for alpha in [0.25, 0.5, 0.75]:
            coords = alpha * fold_a_coords + (1 - alpha) * fold_b_coords
            if fold_a_torsions is not None and fold_b_torsions is not None:
                torsions = _circular_interpolate(
                    fold_a_torsions, fold_b_torsions, alpha
                )
            else:
                torsions = np.zeros((L, 2))
            structures.append(GeneratedStructure(
                ca_coords=coords, torsions=torsions,
                source="perturbed", score=0.8, template_pdb=None,
            ))

        # Random mixed assignments (key for fold-switching)
        n_mixed = min(self.num_candidates // 2, 16)
        for _ in range(n_mixed):
            # Random per-residue fold assignment
            assignment = self.rng.randint(0, 2, size=L)
            coords = np.where(
                assignment[:, None] == 0,
                fold_a_coords, fold_b_coords,
            )
            if fold_a_torsions is not None and fold_b_torsions is not None:
                torsions = np.where(
                    assignment[:, None] == 0,
                    fold_a_torsions, fold_b_torsions,
                )
            else:
                torsions = np.zeros((L, 2))

            # Add small noise
            coords += self.rng.randn(L, 3) * 0.5
            structures.append(GeneratedStructure(
                ca_coords=coords, torsions=torsions,
                source="perturbed", score=0.6, template_pdb=None,
            ))

        # Block-switched assignments (contiguous fold regions)
        for _ in range(min(n_mixed, 8)):
            assignment = np.zeros(L, dtype=int)
            switch_point = self.rng.randint(L // 4, 3 * L // 4)
            assignment[switch_point:] = 1
            coords = np.where(
                assignment[:, None] == 0,
                fold_a_coords, fold_b_coords,
            )
            if fold_a_torsions is not None and fold_b_torsions is not None:
                torsions = np.where(
                    assignment[:, None] == 0,
                    fold_a_torsions, fold_b_torsions,
                )
            else:
                torsions = np.zeros((L, 2))
            structures.append(GeneratedStructure(
                ca_coords=coords, torsions=torsions,
                source="perturbed", score=0.5, template_pdb=None,
            ))

        return structures

    def _perturb_template(
        self,
        template_coords: np.ndarray,
        template_torsions: Optional[np.ndarray],
        num_perturbations: int = 8,
    ) -> List[GeneratedStructure]:
        """Generate structures by perturbing a template."""
        L = len(template_coords)
        structures = []

        for scale in np.linspace(0.2, 2.0, num_perturbations):
            noise = self.rng.randn(L, 3) * self.perturbation_scale * scale
            coords = template_coords + noise

            if template_torsions is not None:
                torsion_noise = self.rng.randn(L, 2) * scale * 0.3
                torsions = template_torsions + torsion_noise
                # Wrap to [-pi, pi]
                torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi
            else:
                torsions = np.zeros((L, 2))

            structures.append(GeneratedStructure(
                ca_coords=coords, torsions=torsions,
                source="perturbed", score=0.5, template_pdb=None,
            ))

        return structures

    def _sample_torsion_space(
        self,
        length: int,
        num_samples: int,
        reference_torsions: Optional[np.ndarray] = None,
    ) -> List[GeneratedStructure]:
        """Generate structures by sampling torsion angle space.
        
        Uses ideal backbone geometry to reconstruct coordinates
        from sampled phi/psi angles.
        """
        structures = []

        for _ in range(num_samples):
            if reference_torsions is not None:
                # Sample around reference with Gaussian noise
                noise = self.rng.randn(length, 2) * self.noise_scale
                torsions = reference_torsions + noise
                torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi
            else:
                # Sample from Ramachandran distribution
                torsions = self._sample_ramachandran(length)

            # Reconstruct backbone from torsion angles
            coords = _torsions_to_coords(torsions, length)

            structures.append(GeneratedStructure(
                ca_coords=coords, torsions=torsions,
                source="fragment", score=0.3, template_pdb=None,
            ))

        return structures

    def _sample_ramachandran(self, length: int) -> np.ndarray:
        """Sample from approximate Ramachandran distribution."""
        torsions = np.zeros((length, 2))
        for i in range(length):
            region = self.rng.choice(["alpha", "beta", "ppII", "random"],
                                      p=[0.4, 0.3, 0.15, 0.15])
            if region == "alpha":
                torsions[i, 0] = np.radians(-60 + self.rng.randn() * 15)
                torsions[i, 1] = np.radians(-47 + self.rng.randn() * 15)
            elif region == "beta":
                torsions[i, 0] = np.radians(-120 + self.rng.randn() * 20)
                torsions[i, 1] = np.radians(130 + self.rng.randn() * 20)
            elif region == "ppII":
                torsions[i, 0] = np.radians(-75 + self.rng.randn() * 10)
                torsions[i, 1] = np.radians(150 + self.rng.randn() * 10)
            else:
                torsions[i, 0] = self.rng.uniform(-np.pi, np.pi)
                torsions[i, 1] = self.rng.uniform(-np.pi, np.pi)
        return torsions

    def _enforce_diversity(
        self, structures: List[GeneratedStructure]
    ) -> List[GeneratedStructure]:
        """Greedy filtering to enforce minimum pairwise diversity."""
        if len(structures) <= 1:
            return structures

        selected = [structures[0]]
        for s in structures[1:]:
            is_diverse = True
            for sel in selected:
                rmsd = np.sqrt(np.mean(
                    np.sum((s.ca_coords - sel.ca_coords) ** 2, axis=1)
                ))
                if rmsd < self.min_diversity_rmsd:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(s)

        return selected


def _circular_interpolate(
    angles_a: np.ndarray,
    angles_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate angles on the circle."""
    diff = angles_b - angles_a
    # Wrap to [-pi, pi]
    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
    result = angles_a + alpha * diff
    return ((result + np.pi) % (2 * np.pi)) - np.pi


def _torsions_to_coords(torsions: np.ndarray, length: int) -> np.ndarray:
    """Reconstruct approximate CA coordinates from torsion angles.
    
    Uses ideal backbone geometry:
    - CA-CA virtual bond length ≈ 3.8 Å
    - The direction changes based on phi/psi angles
    """
    CA_DIST = 3.80  # Angstroms
    coords = np.zeros((length, 3))
    direction = np.array([1.0, 0.0, 0.0])

    for i in range(1, length):
        phi = torsions[i, 0]
        psi = torsions[i - 1, 1] if i > 0 else 0.0

        # Rotation based on torsion angles
        angle = phi + psi
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        # Small out-of-plane component
        tilt = np.array([
            [1, 0, 0],
            [0, np.cos(0.1 * phi), -np.sin(0.1 * phi)],
            [0, np.sin(0.1 * phi), np.cos(0.1 * phi)],
        ])

        direction = tilt @ rotation @ direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        coords[i] = coords[i - 1] + CA_DIST * direction

    return coords
