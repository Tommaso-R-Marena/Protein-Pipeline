#!/usr/bin/env python3
"""
Demonstrate the quantum variational refinement module on a
manageable-size fold-state assignment problem.

Shows:
  1. QUBO construction from structural data
  2. QAOA optimization
  3. VQE optimization
  4. Classical SA comparison
  5. Exhaustive search (for ground truth)
  6. Energy landscape visualization
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcfold.quantum.qubo import build_fold_switch_qubo
from qcfold.quantum.circuits import QAOAFoldOptimizer, VQEFoldOptimizer
from qcfold.quantum.classical_fallback import (
    simulated_annealing, greedy_local_search, exhaustive_search,
)


def generate_small_fold_switch(n_residues=10, seed=42):
    """Generate a synthetic fold-switching region for testing."""
    rng = np.random.RandomState(seed)
    ca_dist = 3.8

    # Fold A: alpha helix
    fold_a_coords = np.zeros((n_residues, 3))
    for i in range(1, n_residues):
        angle = i * 100 * np.pi / 180
        fold_a_coords[i] = fold_a_coords[i-1] + ca_dist * np.array([
            np.cos(angle) * 0.5,
            np.sin(angle) * 0.5,
            1.5,
        ])
    fold_a_torsions = np.column_stack([
        np.full(n_residues, np.radians(-60)) + rng.randn(n_residues) * 0.1,
        np.full(n_residues, np.radians(-47)) + rng.randn(n_residues) * 0.1,
    ])

    # Fold B: beta sheet
    fold_b_coords = np.zeros((n_residues, 3))
    for i in range(1, n_residues):
        fold_b_coords[i] = fold_b_coords[i-1] + ca_dist * np.array([
            (-1)**i * 1.5, 0.3, 3.2,
        ])
    fold_b_torsions = np.column_stack([
        np.full(n_residues, np.radians(-120)) + rng.randn(n_residues) * 0.15,
        np.full(n_residues, np.radians(130)) + rng.randn(n_residues) * 0.15,
    ])

    return fold_a_coords, fold_b_coords, fold_a_torsions, fold_b_torsions


def main():
    output_dir = Path("outputs/quantum_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for n_qubits in [6, 8, 10, 12]:
        print(f"\n{'='*60}")
        print(f"Fold-state assignment: {n_qubits} residues ({n_qubits} qubits)")
        print(f"{'='*60}")

        fold_a_coords, fold_b_coords, fold_a_tor, fold_b_tor = \
            generate_small_fold_switch(n_qubits)

        residue_indices = np.arange(n_qubits)

        # Build QUBO
        qubo = build_fold_switch_qubo(
            fold_a_coords, fold_b_coords,
            fold_a_tor, fold_b_tor,
            residue_indices,
        )

        print(f"QUBO matrix shape: {qubo.Q.shape}")
        print(f"Non-zero entries: {np.count_nonzero(qubo.Q)}")

        # 1. Exhaustive search (ground truth)
        t0 = time.time()
        exact_x, exact_e, exact_hist = exhaustive_search(qubo)
        t_exact = time.time() - t0
        print(f"\nExhaustive search:")
        print(f"  Best energy: {exact_e:.4f}")
        print(f"  Assignment: {exact_x.astype(int)}")
        print(f"  Time: {t_exact:.3f}s")
        print(f"  Energy landscape: min={exact_hist['energy_landscape']['min']:.4f}, "
              f"max={exact_hist['energy_landscape']['max']:.4f}, "
              f"mean={exact_hist['energy_landscape']['mean']:.4f}")

        # 2. QAOA
        print(f"\nQAOA (p={4} layers):")
        t0 = time.time()
        try:
            qaoa_opt = QAOAFoldOptimizer(
                qubo, num_layers=4, backend="default.qubit", seed=42,
            )
            qaoa_x, qaoa_e, qaoa_hist = qaoa_opt.optimize(
                max_iterations=150, lr=0.02, verbose=False,
            )
            t_qaoa = time.time() - t0
            qaoa_optimal = np.allclose(qaoa_x, exact_x) or qaoa_e <= exact_e + 1e-6
            print(f"  Energy: {qaoa_e:.4f} (optimal: {qaoa_optimal})")
            print(f"  Assignment: {qaoa_x}")
            print(f"  Steps: {qaoa_hist['num_steps']}")
            print(f"  Time: {t_qaoa:.3f}s")
            if 'classical_energy' in qaoa_hist:
                print(f"  Classical comparison: {qaoa_hist['classical_energy']:.4f}")
        except Exception as e:
            print(f"  QAOA failed: {e}")
            qaoa_e, t_qaoa, qaoa_optimal = float("inf"), 0, False
            qaoa_x = np.zeros(n_qubits)
            qaoa_hist = {"energies": []}

        # 3. VQE
        print(f"\nVQE (depth=6):")
        t0 = time.time()
        try:
            vqe_opt = VQEFoldOptimizer(
                qubo, circuit_depth=6, backend="default.qubit", seed=42,
            )
            vqe_x, vqe_e, vqe_hist = vqe_opt.optimize(
                max_iterations=150, lr=0.02, verbose=False,
            )
            t_vqe = time.time() - t0
            vqe_optimal = np.allclose(vqe_x, exact_x) or vqe_e <= exact_e + 1e-6
            print(f"  Energy: {vqe_e:.4f} (optimal: {vqe_optimal})")
            print(f"  Assignment: {vqe_x}")
            print(f"  Time: {t_vqe:.3f}s")
        except Exception as e:
            print(f"  VQE failed: {e}")
            vqe_e, t_vqe, vqe_optimal = float("inf"), 0, False
            vqe_hist = {"energies": []}

        # 4. Simulated Annealing
        print(f"\nSimulated Annealing:")
        t0 = time.time()
        sa_x, sa_e, sa_hist = simulated_annealing(qubo, num_restarts=10, seed=42)
        t_sa = time.time() - t0
        sa_optimal = np.allclose(sa_x, exact_x) or sa_e <= exact_e + 1e-6
        print(f"  Energy: {sa_e:.4f} (optimal: {sa_optimal})")
        print(f"  Assignment: {sa_x.astype(int)}")
        print(f"  Time: {t_sa:.3f}s")

        # 5. Greedy
        print(f"\nGreedy local search:")
        t0 = time.time()
        greedy_x, greedy_e, _ = greedy_local_search(qubo, seed=42)
        t_greedy = time.time() - t0
        greedy_optimal = np.allclose(greedy_x, exact_x) or greedy_e <= exact_e + 1e-6
        print(f"  Energy: {greedy_e:.4f} (optimal: {greedy_optimal})")
        print(f"  Time: {t_greedy:.3f}s")

        results[n_qubits] = {
            "exact_energy": float(exact_e),
            "qaoa_energy": float(qaoa_e),
            "qaoa_optimal": bool(qaoa_optimal),
            "qaoa_time": t_qaoa,
            "qaoa_assignment": qaoa_x.tolist(),
            "qaoa_convergence": [float(e) for e in qaoa_hist.get("energies", [])],
            "vqe_energy": float(vqe_e),
            "vqe_optimal": bool(vqe_optimal),
            "vqe_time": t_vqe,
            "vqe_convergence": [float(e) for e in vqe_hist.get("energies", [])],
            "sa_energy": float(sa_e),
            "sa_optimal": bool(sa_optimal),
            "sa_time": t_sa,
            "greedy_energy": float(greedy_e),
            "greedy_optimal": bool(greedy_optimal),
            "greedy_time": t_greedy,
            "energy_landscape": exact_hist["energy_landscape"],
        }

    # Summary table
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'N':>4} {'Exact':>10} {'QAOA':>10} {'VQE':>10} "
          f"{'SA':>10} {'Greedy':>10}")
    print("-" * 60)
    for n, r in results.items():
        print(f"{n:>4} {r['exact_energy']:>10.4f} "
              f"{r['qaoa_energy']:>10.4f}{'*' if r['qaoa_optimal'] else ' '} "
              f"{r['vqe_energy']:>10.4f}{'*' if r['vqe_optimal'] else ' '} "
              f"{r['sa_energy']:>10.4f}{'*' if r['sa_optimal'] else ' '} "
              f"{r['greedy_energy']:>10.4f}{'*' if r['greedy_optimal'] else ' '}")
    print("(* = found optimal solution)")

    print(f"\n{'N':>4} {'QAOA t':>10} {'VQE t':>10} {'SA t':>10} {'Greedy t':>10}")
    print("-" * 50)
    for n, r in results.items():
        print(f"{n:>4} {r['qaoa_time']:>9.3f}s {r['vqe_time']:>9.3f}s "
              f"{r['sa_time']:>9.3f}s {r['greedy_time']:>9.3f}s")

    # Save results
    with open(output_dir / "quantum_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/quantum_demo_results.json")
    return results


if __name__ == "__main__":
    results = main()
