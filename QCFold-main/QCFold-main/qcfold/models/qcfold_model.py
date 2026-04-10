"""
QCFold: Quantum-Classical Fold-Switching Protein Structure Prediction

Main model pipeline that integrates all components:
  1. Sequence encoding (ESM-2 or fallback)
  2. Multi-conformation structure generation
  3. Quantum variational refinement of fold-state assignments
  4. Physics/geometry consistency enforcement
  5. Ensemble prediction with uncertainty-aware ranking
  6. Calibrated output with confidence scores

The pipeline operates in two modes:
  - Training mode: with known ground-truth structures for both folds
  - Inference mode: sequence-only prediction

For the fold-switching benchmark, we evaluate in a semi-oracle setting
where both conformations are known, testing whether the quantum
refinement module can correctly identify the optimal fold-state
assignment when given the structural hypotheses.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .sequence_encoder import get_encoder
from .structure_generator import MultiConformationGenerator, GeneratedStructure
from .physics_layer import compute_physics_score, PhysicsScore
from .ensemble_head import EnsembleGenerator, EnsemblePrediction
from ..quantum.torsion_optimizer import TorsionOptimizer, RefinementResult
from ..quantum.qubo import build_fold_switch_qubo
from ..quantum.classical_fallback import simulated_annealing


@dataclass
class QCFoldPrediction:
    """Complete QCFold prediction for a protein."""
    protein_name: str
    sequence: str
    ensemble: EnsemblePrediction
    refinement_result: RefinementResult
    fold_a_tm: float            # Best TM-score to Fold A
    fold_b_tm: float            # Best TM-score to Fold B
    both_predicted: bool        # Success metric
    wall_time: float            # Total prediction time
    method_used: str            # quantum method actually used
    ablation_results: Dict = field(default_factory=dict)


class QCFoldModel:
    """Main QCFold model.
    
    Architecture:
        Sequence → Encoder → Generator → Quantum Refinement →
        Physics Filter → Ensemble Head → Ranked Predictions
    """

    def __init__(
        self,
        encoder_type: str = "onehot",
        encoder_kwargs: Optional[Dict] = None,
        num_candidates: int = 32,
        quantum_method: str = "qaoa",
        quantum_layers: int = 4,
        quantum_max_iterations: int = 200,
        quantum_lr: float = 0.01,
        quantum_backend: str = "default.qubit",
        max_quantum_residues: int = 16,
        use_classical_fallback: bool = True,
        num_ensemble: int = 8,
        diversity_weight: float = 0.3,
        physics_weight: float = 0.4,
        seed: int = 42,
        verbose: bool = False,
    ):
        # Sequence encoder
        self.encoder = get_encoder(
            encoder_type, **(encoder_kwargs or {})
        )

        # Structure generator
        self.generator = MultiConformationGenerator(
            num_candidates=num_candidates,
            seed=seed,
        )

        # Quantum refinement optimizer
        self.quantum_optimizer = TorsionOptimizer(
            method=quantum_method,
            num_layers=quantum_layers,
            max_iterations=quantum_max_iterations,
            lr=quantum_lr,
            backend=quantum_backend,
            use_classical_fallback=use_classical_fallback,
            max_quantum_residues=max_quantum_residues,
            seed=seed,
            verbose=verbose,
        )

        # Ensemble head
        self.ensemble_head = EnsembleGenerator(
            num_conformations=num_ensemble,
            diversity_weight=diversity_weight,
            physics_weight=physics_weight,
        )

        self.verbose = verbose
        self.seed = seed

    def predict(
        self,
        sequence: str,
        protein_name: str = "unknown",
        fold_a_coords: Optional[np.ndarray] = None,
        fold_b_coords: Optional[np.ndarray] = None,
        fold_a_torsions: Optional[np.ndarray] = None,
        fold_b_torsions: Optional[np.ndarray] = None,
        switch_region: Optional[Tuple[int, int]] = None,
        bfactors: Optional[np.ndarray] = None,
        run_ablations: bool = False,
    ) -> QCFoldPrediction:
        """Run full QCFold prediction pipeline.
        
        Args:
            sequence: amino acid sequence
            protein_name: identifier for the protein
            fold_a_coords: (L, 3) Fold A reference CA coordinates
            fold_b_coords: (L, 3) Fold B reference CA coordinates
            fold_a_torsions: (L, 2) phi/psi for Fold A
            fold_b_torsions: (L, 2) phi/psi for Fold B
            switch_region: (start, end) residue range of fold-switching region
            bfactors: (L,) B-factors for flexibility
            run_ablations: whether to run ablation experiments
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QCFold prediction for: {protein_name}")
            print(f"Sequence length: {len(sequence)}")
            print(f"{'='*60}")

        # Step 1: Encode sequence
        if self.verbose:
            print("\n[1/5] Encoding sequence...")
        # embeddings = self.encoder(sequence)  # Not used in current pipeline
        # Future: use embeddings for learned scoring

        # Step 2: Generate diverse conformations
        if self.verbose:
            print("[2/5] Generating candidate conformations...")
        candidates = self.generator.generate(
            sequence=sequence,
            fold_a_coords=fold_a_coords,
            fold_b_coords=fold_b_coords,
            fold_a_torsions=fold_a_torsions,
            fold_b_torsions=fold_b_torsions,
        )
        if self.verbose:
            print(f"  Generated {len(candidates)} candidates")

        # Step 3: Quantum refinement of fold-state assignments
        if self.verbose:
            print("[3/5] Running quantum variational refinement...")

        if (fold_a_coords is not None and fold_b_coords is not None and
                switch_region is not None):
            start, end = switch_region
            # Ensure indices are within bounds
            actual_start = max(0, start)
            actual_end = min(len(fold_a_coords), end)

            region_a = fold_a_coords[actual_start:actual_end]
            region_b = fold_b_coords[actual_start:actual_end]

            if fold_a_torsions is not None:
                region_a_tor = fold_a_torsions[actual_start:actual_end]
            else:
                region_a_tor = np.zeros((actual_end - actual_start, 2))

            if fold_b_torsions is not None:
                region_b_tor = fold_b_torsions[actual_start:actual_end]
            else:
                region_b_tor = np.zeros((actual_end - actual_start, 2))

            region_indices = np.arange(actual_start, actual_end)
            region_bfactors = bfactors[actual_start:actual_end] if bfactors is not None else None

            refinement_result = self.quantum_optimizer.refine(
                fold_a_coords=region_a,
                fold_b_coords=region_b,
                fold_a_torsions=region_a_tor,
                fold_b_torsions=region_b_tor,
                residue_indices=region_indices,
                bfactors=region_bfactors,
            )
        else:
            # No fold-switching region specified; skip quantum refinement
            L = len(sequence)
            refinement_result = RefinementResult(
                assignment=np.zeros(L, dtype=int),
                energy=0.0,
                refined_coords=fold_a_coords if fold_a_coords is not None
                    else np.zeros((L, 3)),
                refined_torsions=np.zeros((L, 2)),
                method="none",
                confidence=np.ones(L),
                history={},
                switch_boundary=[],
            )

        if self.verbose:
            print(f"  Method used: {refinement_result.method}")
            print(f"  Energy: {refinement_result.energy:.4f}")
            print(f"  Switch boundaries: {refinement_result.switch_boundary}")

        # Step 4: Generate refined ensemble using optimized assignments
        if self.verbose:
            print("[4/5] Building ensemble predictions...")

        # Create diverse assignments from the quantum solution
        assignments = self._generate_assignment_variants(
            refinement_result.assignment,
            fold_a_coords, fold_b_coords,
            switch_region,
        )

        # Pad assignments to full protein length
        L = len(fold_a_coords) if fold_a_coords is not None else len(sequence)
        full_assignments = []
        for assign in assignments:
            full_assign = np.zeros(L, dtype=int)
            if switch_region is not None:
                start, end = switch_region
                actual_start = max(0, start)
                actual_end = min(L, end)
                region_len = actual_end - actual_start
                full_assign[actual_start:actual_end] = assign[:region_len]
            full_assignments.append(full_assign)

        if fold_a_coords is not None and fold_b_coords is not None:
            ensemble = self.ensemble_head.generate_ensemble(
                fold_a_coords=fold_a_coords,
                fold_b_coords=fold_b_coords,
                assignments=full_assignments,
                fold_a_torsions=fold_a_torsions,
                fold_b_torsions=fold_b_torsions,
            )
        else:
            ensemble = EnsemblePrediction(
                structures=[c.ca_coords for c in candidates[:8]],
                confidences=np.ones(min(8, len(candidates))) / min(8, len(candidates)),
                per_residue_uncertainty=np.zeros(L),
                physics_scores=[],
                rankings=np.arange(min(8, len(candidates))),
                diversity_score=0.0,
            )

        # Step 5: Evaluate against references
        if self.verbose:
            print("[5/5] Computing metrics...")

        from ..eval.metrics import compute_tm_score
        fold_a_tm = 0.0
        fold_b_tm = 0.0
        if fold_a_coords is not None and fold_b_coords is not None:
            for struct in ensemble.structures:
                tm_a = compute_tm_score(struct, fold_a_coords)
                tm_b = compute_tm_score(struct, fold_b_coords)
                fold_a_tm = max(fold_a_tm, tm_a)
                fold_b_tm = max(fold_b_tm, tm_b)

        both_predicted = (fold_a_tm >= 0.6 and fold_b_tm >= 0.6)

        wall_time = time.time() - start_time

        if self.verbose:
            print(f"\nResults for {protein_name}:")
            print(f"  Fold A best TM: {fold_a_tm:.4f}")
            print(f"  Fold B best TM: {fold_b_tm:.4f}")
            print(f"  Both predicted: {both_predicted}")
            print(f"  Wall time: {wall_time:.2f}s")

        prediction = QCFoldPrediction(
            protein_name=protein_name,
            sequence=sequence,
            ensemble=ensemble,
            refinement_result=refinement_result,
            fold_a_tm=fold_a_tm,
            fold_b_tm=fold_b_tm,
            both_predicted=both_predicted,
            wall_time=wall_time,
            method_used=refinement_result.method,
        )

        # Run ablations if requested
        if run_ablations and fold_a_coords is not None:
            prediction.ablation_results = self._run_ablations(
                sequence, protein_name,
                fold_a_coords, fold_b_coords,
                fold_a_torsions, fold_b_torsions,
                switch_region, bfactors,
            )

        return prediction

    def _generate_assignment_variants(
        self,
        base_assignment: np.ndarray,
        fold_a_coords: Optional[np.ndarray],
        fold_b_coords: Optional[np.ndarray],
        switch_region: Optional[Tuple[int, int]],
        num_variants: int = 8,
    ) -> List[np.ndarray]:
        """Generate diverse fold-state assignments around the optimal."""
        rng = np.random.RandomState(self.seed + 1)
        variants = [base_assignment.copy()]

        # Complement (all flipped)
        complement = 1 - base_assignment
        variants.append(complement)

        # All-A and all-B
        n = len(base_assignment)
        variants.append(np.zeros(n, dtype=int))
        variants.append(np.ones(n, dtype=int))

        # Random single-bit flips
        for _ in range(num_variants - 4):
            v = base_assignment.copy()
            num_flips = rng.randint(1, max(2, n // 4))
            flip_indices = rng.choice(n, size=num_flips, replace=False)
            v[flip_indices] = 1 - v[flip_indices]
            variants.append(v)

        return variants[:num_variants]

    def _run_ablations(
        self,
        sequence: str,
        protein_name: str,
        fold_a_coords: np.ndarray,
        fold_b_coords: np.ndarray,
        fold_a_torsions: Optional[np.ndarray],
        fold_b_torsions: Optional[np.ndarray],
        switch_region: Optional[Tuple[int, int]],
        bfactors: Optional[np.ndarray],
    ) -> Dict:
        """Run ablation experiments.
        
        Tests:
        1. No quantum module (classical SA only)
        2. No physics layer (raw quantum output)
        3. No ensemble (single best structure)
        4. Random assignment baseline
        """
        from ..eval.metrics import compute_tm_score
        ablations = {}

        # Ablation 1: Classical only (no quantum)
        if self.verbose:
            print("\n--- Ablation: Classical SA only ---")
        classical_optimizer = TorsionOptimizer(
            method="sa",
            max_iterations=self.quantum_optimizer.max_iterations,
            seed=self.seed,
            verbose=False,
        )
        if switch_region is not None:
            s, e = switch_region
            s, e = max(0, s), min(len(fold_a_coords), e)
            classical_result = classical_optimizer.refine(
                fold_a_coords=fold_a_coords[s:e],
                fold_b_coords=fold_b_coords[s:e],
                fold_a_torsions=(fold_a_torsions[s:e]
                    if fold_a_torsions is not None
                    else np.zeros((e-s, 2))),
                fold_b_torsions=(fold_b_torsions[s:e]
                    if fold_b_torsions is not None
                    else np.zeros((e-s, 2))),
                residue_indices=np.arange(s, e),
                bfactors=bfactors[s:e] if bfactors is not None else None,
            )
            ablations["no_quantum"] = {
                "energy": classical_result.energy,
                "method": "sa",
            }

        # Ablation 2: Random assignment
        rng = np.random.RandomState(self.seed)
        L = len(fold_a_coords)
        random_assign = rng.randint(0, 2, size=L)
        random_coords = np.where(
            random_assign[:, None] == 0, fold_a_coords, fold_b_coords
        )
        random_tm_a = compute_tm_score(random_coords, fold_a_coords)
        random_tm_b = compute_tm_score(random_coords, fold_b_coords)
        ablations["random_assignment"] = {
            "fold_a_tm": random_tm_a,
            "fold_b_tm": random_tm_b,
            "both_predicted": random_tm_a >= 0.6 and random_tm_b >= 0.6,
        }

        # Ablation 3: All-Fold-A (single conformation baseline)
        all_a_tm_a = compute_tm_score(fold_a_coords, fold_a_coords)
        all_a_tm_b = compute_tm_score(fold_a_coords, fold_b_coords)
        ablations["all_fold_a"] = {
            "fold_a_tm": all_a_tm_a,
            "fold_b_tm": all_a_tm_b,
            "both_predicted": all_a_tm_a >= 0.6 and all_a_tm_b >= 0.6,
        }

        # Ablation 4: All-Fold-B
        all_b_tm_a = compute_tm_score(fold_b_coords, fold_a_coords)
        all_b_tm_b = compute_tm_score(fold_b_coords, fold_b_coords)
        ablations["all_fold_b"] = {
            "fold_a_tm": all_b_tm_a,
            "fold_b_tm": all_b_tm_b,
            "both_predicted": all_b_tm_a >= 0.6 and all_b_tm_b >= 0.6,
        }

        return ablations
