"""
Classical optimization fallbacks for the fold-state assignment problem.

Implements:
  1. Simulated Annealing (SA)
  2. Greedy with local search
  3. Exhaustive search (for small instances)

These serve as:
  - Baselines for comparison against quantum methods
  - Fallback when quantum simulation is too expensive
  - Upper bounds on solution quality
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .qubo import QUBOInstance


def simulated_annealing(
    qubo: QUBOInstance,
    temperature: float = 2.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 10000,
    num_restarts: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, float, Dict]:
    """Simulated annealing for QUBO optimization.
    
    Returns:
        best_assignment: (n,) binary array
        best_energy: float
        history: dict with optimization trace
    """
    rng = np.random.RandomState(seed)
    n = qubo.n_residues
    Q = qubo.Q

    global_best_x = None
    global_best_e = float("inf")
    all_histories = []

    for restart in range(num_restarts):
        # Random initial assignment
        x = rng.randint(0, 2, size=n)
        current_e = float(x @ Q @ x + qubo.offset)
        best_x = x.copy()
        best_e = current_e
        T = temperature
        energies = [current_e]

        for step in range(max_iterations):
            # Flip a random bit
            i = rng.randint(n)
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]
            new_e = float(x_new @ Q @ x_new + qubo.offset)

            # Metropolis criterion
            delta = new_e - current_e
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
                current_e = new_e

            if current_e < best_e:
                best_e = current_e
                best_x = x.copy()

            T *= cooling_rate
            energies.append(current_e)

        all_histories.append(energies)

        if best_e < global_best_e:
            global_best_e = best_e
            global_best_x = best_x.copy()

    history = {
        "all_restarts": all_histories,
        "num_restarts": num_restarts,
    }

    return global_best_x, global_best_e, history


def greedy_local_search(
    qubo: QUBOInstance,
    max_iterations: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, float, Dict]:
    """Greedy search with random restarts and local improvement."""
    rng = np.random.RandomState(seed)
    n = qubo.n_residues
    Q = qubo.Q

    best_x = None
    best_e = float("inf")

    for _ in range(10):  # 10 random restarts
        x = rng.randint(0, 2, size=n)
        improved = True

        while improved:
            improved = False
            for i in range(n):
                x_flip = x.copy()
                x_flip[i] = 1 - x_flip[i]
                e_orig = float(x @ Q @ x)
                e_flip = float(x_flip @ Q @ x_flip)
                if e_flip < e_orig:
                    x = x_flip
                    improved = True

        e = float(x @ Q @ x + qubo.offset)
        if e < best_e:
            best_e = e
            best_x = x.copy()

    return best_x, best_e, {}


def exhaustive_search(qubo: QUBOInstance) -> Tuple[np.ndarray, float, Dict]:
    """Exhaustive search over all 2^n assignments. Only for n <= 20."""
    n = qubo.n_residues
    if n > 20:
        raise ValueError(f"Exhaustive search infeasible for n={n} > 20")

    Q = qubo.Q
    best_x = None
    best_e = float("inf")
    all_energies = []

    for bits in range(2**n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        e = float(x @ Q @ x + qubo.offset)
        all_energies.append(e)
        if e < best_e:
            best_e = e
            best_x = x.copy()

    history = {
        "all_energies": all_energies,
        "energy_landscape": {
            "min": min(all_energies),
            "max": max(all_energies),
            "mean": np.mean(all_energies),
            "std": np.std(all_energies),
        },
    }

    return best_x, best_e, history
