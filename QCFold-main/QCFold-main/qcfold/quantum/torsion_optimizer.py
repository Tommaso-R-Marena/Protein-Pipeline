"""
Torsion angle optimizer: the high-level interface for quantum-assisted
fold-state refinement.

This module orchestrates:
  1. QUBO construction from protein structural data
  2. Quantum circuit optimization (QAOA or VQE)
  3. Classical fallback optimization
  4. Decoding of optimized fold-state assignments back to coordinates
  5. Hybrid structure reconstruction

The key novel contribution: we use quantum optimization to solve the
discrete combinatorial problem of assigning each residue in a fold-
switching region to one of two conformational states, where the cost
function encodes structural consistency, physics constraints, and
energy landscape information.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from .qubo import QUBOInstance, build_fold_switch_qubo
from .classical_fallback import simulated_annealing, greedy_local_search, exhaustive_search


@dataclass
class RefinementResult:
    """Result of quantum/classical fold-state refinement."""
    assignment: np.ndarray           # (n,) binary: 0=FoldA, 1=FoldB
    energy: float                    # QUBO energy of the assignment
    refined_coords: np.ndarray       # (n, 3) refined CA coordinates
    refined_torsions: np.ndarray     # (n, 2) refined phi/psi
    method: str                      # "qaoa", "vqe", "sa", "greedy", "exhaustive"
    confidence: np.ndarray           # (n,) per-residue confidence [0, 1]
    history: Dict                    # Optimization history
    switch_boundary: List[int]       # Residue indices where fold switches


class TorsionOptimizer:
    """High-level optimizer for fold-state assignment and torsion refinement."""

    def __init__(
        self,
        method: str = "qaoa",
        num_layers: int = 4,
        circuit_depth: int = 6,
        max_iterations: int = 200,
        lr: float = 0.01,
        backend: str = "default.qubit",
        use_classical_fallback: bool = True,
        max_quantum_residues: int = 16,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.method = method
        self.num_layers = num_layers
        self.circuit_depth = circuit_depth
        self.max_iterations = max_iterations
        self.lr = lr
        self.backend = backend
        self.use_classical_fallback = use_classical_fallback
        self.max_quantum_residues = max_quantum_residues
        self.seed = seed
        self.verbose = verbose

    def refine(
        self,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
        fold_a_torsions: np.ndarray,
        fold_b_torsions: np.ndarray,
        residue_indices: np.ndarray,
        bfactors: Optional[np.ndarray] = None,
    ) -> RefinementResult:
        """Run fold-state refinement on a switching region.
        
        Args:
            fold_a_coords: (n, 3) CA coordinates for Fold A
            fold_b_coords: (n, 3) CA coordinates for Fold B
            fold_a_torsions: (n, 2) phi/psi for Fold A
            fold_b_torsions: (n, 2) phi/psi for Fold B
            residue_indices: (n,) original residue numbers
            bfactors: (n,) B-factors for flexibility estimation
            
        Returns:
            RefinementResult with assignment and refined coordinates
        """
        n = len(fold_a_coords)

        # Build QUBO
        qubo = build_fold_switch_qubo(
            fold_a_coords=fold_a_coords,
            fold_b_coords=fold_b_coords,
            fold_a_torsions=fold_a_torsions,
            fold_b_torsions=fold_b_torsions,
            residue_indices=residue_indices,
            boundary_flexibility=bfactors,
        )

        # Choose optimization method
        if n <= self.max_quantum_residues and self.method in ("qaoa", "vqe"):
            assignment, energy, history = self._quantum_optimize(qubo)
            method_used = self.method
        elif n <= 20 and self.method == "exhaustive":
            assignment, energy, history = exhaustive_search(qubo)
            method_used = "exhaustive"
        else:
            # Classical fallback
            assignment, energy, history = simulated_annealing(
                qubo, max_iterations=self.max_iterations * 10,
                seed=self.seed,
            )
            method_used = "sa"

        # If quantum was attempted, also run classical for comparison
        if method_used in ("qaoa", "vqe") and self.use_classical_fallback:
            sa_assign, sa_energy, sa_hist = simulated_annealing(
                qubo, seed=self.seed
            )
            if sa_energy < energy:
                if self.verbose:
                    print(f"  Classical SA found better solution: "
                          f"{sa_energy:.4f} vs {energy:.4f}")
                # Keep quantum solution but note the discrepancy
                history["classical_better"] = True
                history["classical_energy"] = sa_energy
            else:
                history["classical_better"] = False
                history["classical_energy"] = sa_energy

        # Decode assignment to coordinates and torsions
        refined_coords = self._decode_coordinates(
            assignment, fold_a_coords, fold_b_coords
        )
        refined_torsions = self._decode_torsions(
            assignment, fold_a_torsions, fold_b_torsions
        )

        # Compute per-residue confidence
        confidence = self._compute_confidence(
            assignment, qubo, fold_a_coords, fold_b_coords
        )

        # Find fold-switch boundaries
        switch_boundary = self._find_boundaries(assignment, residue_indices)

        return RefinementResult(
            assignment=assignment,
            energy=energy,
            refined_coords=refined_coords,
            refined_torsions=refined_torsions,
            method=method_used,
            confidence=confidence,
            history=history,
            switch_boundary=switch_boundary,
        )

    def _quantum_optimize(
        self, qubo: QUBOInstance
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run quantum optimization (QAOA or VQE)."""
        if self.method == "qaoa":
            from .circuits import QAOAFoldOptimizer
            optimizer = QAOAFoldOptimizer(
                qubo=qubo,
                num_layers=self.num_layers,
                backend=self.backend,
                seed=self.seed,
            )
        else:
            from .circuits import VQEFoldOptimizer
            optimizer = VQEFoldOptimizer(
                qubo=qubo,
                circuit_depth=self.circuit_depth,
                backend=self.backend,
                seed=self.seed,
            )
        return optimizer.optimize(
            max_iterations=self.max_iterations,
            lr=self.lr,
            verbose=self.verbose,
        )

    def _decode_coordinates(
        self,
        assignment: np.ndarray,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
    ) -> np.ndarray:
        """Decode binary assignment to 3D coordinates.
        
        For each residue, select Fold A or Fold B coordinates based
        on the assignment, then smooth the boundaries to avoid
        discontinuities.
        """
        n = len(assignment)
        coords = np.zeros_like(fold_a_coords)

        for i in range(n):
            if assignment[i] == 0:
                coords[i] = fold_a_coords[i]
            else:
                coords[i] = fold_b_coords[i]

        # Smooth boundaries: interpolate at fold-switch boundaries
        # to avoid backbone discontinuities
        for i in range(1, n):
            if assignment[i] != assignment[i - 1]:
                # Boundary detected: blend coordinates
                alpha = 0.5  # Equal blend at boundary
                coords[i] = alpha * fold_a_coords[i] + (1 - alpha) * fold_b_coords[i]
                if i > 0:
                    coords[i - 1] = (
                        alpha * fold_a_coords[i - 1] +
                        (1 - alpha) * fold_b_coords[i - 1]
                    )

        return coords

    def _decode_torsions(
        self,
        assignment: np.ndarray,
        fold_a_torsions: np.ndarray,
        fold_b_torsions: np.ndarray,
    ) -> np.ndarray:
        """Decode binary assignment to torsion angles."""
        n = len(assignment)
        torsions = np.zeros_like(fold_a_torsions)
        for i in range(n):
            if assignment[i] == 0:
                torsions[i] = fold_a_torsions[i]
            else:
                torsions[i] = fold_b_torsions[i]
        return torsions

    def _compute_confidence(
        self,
        assignment: np.ndarray,
        qubo: QUBOInstance,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
    ) -> np.ndarray:
        """Compute per-residue confidence for the assignment.
        
        Confidence is based on:
        1. Energy gap between current and flipped assignment at each residue
        2. Local structural consistency
        """
        n = len(assignment)
        confidence = np.zeros(n)

        for i in range(n):
            # Flip residue i and compute energy change
            flipped = assignment.copy()
            flipped[i] = 1 - flipped[i]
            e_current = qubo.evaluate(assignment)
            e_flipped = qubo.evaluate(flipped)
            energy_gap = e_flipped - e_current

            # High gap → high confidence (flipping makes it worse)
            # Normalize with sigmoid
            confidence[i] = 1.0 / (1.0 + np.exp(-energy_gap))

        return confidence

    def _find_boundaries(
        self, assignment: np.ndarray, residue_indices: np.ndarray
    ) -> List[int]:
        """Find residue indices where fold state changes."""
        boundaries = []
        for i in range(1, len(assignment)):
            if assignment[i] != assignment[i - 1]:
                boundaries.append(int(residue_indices[i]))
        return boundaries


def multi_region_refine(
    optimizer: TorsionOptimizer,
    regions: List[Dict],
) -> List[RefinementResult]:
    """Refine multiple fold-switching regions independently.
    
    Args:
        optimizer: TorsionOptimizer instance
        regions: list of dicts with keys:
            fold_a_coords, fold_b_coords, fold_a_torsions,
            fold_b_torsions, residue_indices, bfactors (optional)
    """
    results = []
    for region in regions:
        result = optimizer.refine(**region)
        results.append(result)
    return results
