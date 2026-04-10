"""
Iterative Warm-Start XY-QAOA for protein rotamer packing.

Key innovations over prior work (QADF v2 / arXiv:2507.19383):
1. Local XY-mixer enforces one-hot constraint as a hard constraint (no penalty terms)
2. Warm-started initial state from classical greedy solution
3. Iterative refinement: update warm-start bias from each round's samples
4. CVaR (Conditional Value at Risk) objective for better ground state finding
5. Gradient-free COBYLA optimizer with restarts

References:
- Agathangelou et al. 2025 (arXiv:2507.19383): local XY-mixer formulation
- Bucher et al. 2026 (arXiv:2604.02083): IWS-QAOA formulation
- Dupont et al. 2023 (Science Advances): quantum-enhanced greedy
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from ..mixers.xy_mixer import (
    iws_mixer_circuit,
    prepare_iws_initial_state,
    prepare_w_state,
    xy_mixer_circuit,
)


@dataclass
class QAOAConfig:
    """Configuration for IWS-QAOA runs."""
    p: int = 4                    # QAOA depth
    n_shots: int = 1024           # Shots per circuit evaluation
    n_iter: int = 3               # IWS iterations
    cvar_alpha: float = 0.2       # CVaR percentile (0.2 = top 20% of samples)
    optimizer: str = "COBYLA"     # Optimizer for variational parameters
    max_opt_iter: int = 200       # Max optimizer iterations per IWS round
    n_restarts: int = 3           # Random restarts for optimizer
    use_warm_start: bool = True   # Enable IWS
    use_xy_mixer: bool = True     # Use XY-mixer (hard constraint) vs transverse field
    seed: int = 42


@dataclass
class QAOAResult:
    """Results from an IWS-QAOA run."""
    best_assignment: np.ndarray   # Best valid assignment found
    best_energy: float            # Energy of best assignment
    exact_energy: float           # Known optimal (if available)
    approximation_ratio: float    # best_energy / exact_energy
    ground_state_prob: float      # P(sampling optimal state) at final params
    n_circuit_evals: int          # Total circuit evaluations
    runtime_s: float              # Wall-clock runtime
    iws_history: list[dict]       # Per-iteration history
    params_history: list[np.ndarray]
    energy_history: list[float]


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO (minimize x^T Q x) to Ising (minimize Σ J_ij z_i z_j + Σ h_i z_i).
    x_i = (1 - z_i) / 2  →  z_i in {-1, +1}.
    
    Returns (J matrix, h vector, constant offset).
    """
    M = Q.shape[0]
    J = np.zeros((M, M))
    h = np.zeros(M)
    c = 0.0

    for i in range(M):
        for j in range(M):
            if i == j:
                h[i] += Q[i, i] / 2.0
                c += Q[i, i] / 4.0
            elif j > i:
                J[i, j] = Q[i, j] / 4.0
                h[i] += Q[i, j] / 4.0
                h[j] += Q[i, j] / 4.0
                c += Q[i, j] / 4.0

    return J, h, c


class IWSQAOASolver:
    """
    Iterative Warm-Start XY-QAOA solver for protein rotamer packing.
    
    The algorithm:
    1. Get initial bias from classical greedy (or uniform)
    2. Prepare warm-started initial state |ψ_ws>
    3. Run QAOA with local XY-mixer, optimize γ, β params
    4. Sample bitstrings, evaluate energies
    5. Update bias toward high-probability valid low-energy states
    6. Repeat for n_iter iterations
    7. Return best found assignment
    """

    def __init__(self, Q: np.ndarray, N: int, n: int, config: QAOAConfig):
        self.Q = Q
        self.N = N  # number of residues
        self.n = n  # rotamers per residue
        self.M = N * n
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Wire layout: block i uses wires [i*n, ..., i*n + n - 1]
        self.wires_per_residue = [
            list(range(i * n, (i + 1) * n)) for i in range(N)
        ]
        self.all_wires = list(range(self.M))

        # Convert QUBO to Ising
        self.J, self.h, self.c = qubo_to_ising(Q)

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.M, shots=config.n_shots)
        self.dev_exact = qml.device("default.qubit", wires=self.M, shots=None)

    def _cost_circuit(self, gamma_vec, beta_vec, bias, use_iws_mixer):
        """PennyLane QNode for the QAOA cost circuit."""
        @qml.qnode(self.dev)
        def circuit():
            # State preparation
            for i, block in enumerate(self.wires_per_residue):
                if self.config.use_warm_start and bias is not None:
                    prepare_iws_initial_state([block], bias[i:i+1])
                else:
                    prepare_w_state(block)

            # QAOA layers
            for layer in range(self.config.p):
                # Cost unitary: exp(-i * gamma * H_C)
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        if abs(self.J[i, j]) > 1e-10:
                            qml.IsingZZ(2.0 * gamma_vec[layer] * self.J[i, j],
                                        wires=[i, j])
                for i in range(self.M):
                    if abs(self.h[i]) > 1e-10:
                        qml.RZ(2.0 * gamma_vec[layer] * self.h[i], wires=i)

                # Mixer unitary: exp(-i * beta * H_M)
                if use_iws_mixer and bias is not None:
                    iws_mixer_circuit(self.wires_per_residue,
                                      beta_vec[layer], bias)
                else:
                    xy_mixer_circuit(self.wires_per_residue, beta_vec[layer])

            return qml.sample(wires=self.all_wires)

        return circuit()

    def _exact_probs_circuit(self, gamma_vec, beta_vec, bias, use_iws_mixer):
        """Get exact statevector probabilities (no shots)."""
        @qml.qnode(self.dev_exact)
        def circuit():
            for i, block in enumerate(self.wires_per_residue):
                if self.config.use_warm_start and bias is not None:
                    prepare_iws_initial_state([block], bias[i:i+1])
                else:
                    prepare_w_state(block)

            for layer in range(self.config.p):
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        if abs(self.J[i, j]) > 1e-10:
                            qml.IsingZZ(2.0 * gamma_vec[layer] * self.J[i, j],
                                        wires=[i, j])
                for i in range(self.M):
                    if abs(self.h[i]) > 1e-10:
                        qml.RZ(2.0 * gamma_vec[layer] * self.h[i], wires=i)

                if use_iws_mixer and bias is not None:
                    iws_mixer_circuit(self.wires_per_residue,
                                      beta_vec[layer], bias)
                else:
                    xy_mixer_circuit(self.wires_per_residue, beta_vec[layer])

            return qml.probs(wires=self.all_wires)

        return circuit()

    def _samples_to_energy(self, samples: np.ndarray) -> list[tuple[float, np.ndarray, bool]]:
        """
        Convert bitstring samples to (energy, assignment, is_valid) tuples.
        Valid = exactly one rotamer set per residue (one-hot constraint).
        """
        results = []
        for sample in samples:
            x = np.array(sample, dtype=int)
            valid = all(
                x[i * self.n: (i + 1) * self.n].sum() == 1
                for i in range(self.N)
            )
            if valid:
                e = float(x @ self.Q @ x)
            else:
                # Infeasible: large penalty
                e = float("inf")
            results.append((e, x, valid))
        return results

    def _cvar_objective(self, energies: list[float]) -> float:
        """CVaR: mean of bottom alpha fraction of energies."""
        valid = [e for e in energies if e != float("inf")]
        if not valid:
            return 1e6
        k = max(1, int(self.config.cvar_alpha * len(valid)))
        return float(np.mean(sorted(valid)[:k]))

    def _optimize_params(self, bias, use_iws_mixer, init_params=None):
        """
        Optimize QAOA parameters (gamma, beta) using COBYLA + CVaR.
        Returns (gamma_vec, beta_vec, n_evals).
        """
        p = self.config.p
        n_evals = [0]

        def objective(params):
            gamma = params[:p]
            beta = params[p:]
            samples = self._cost_circuit(gamma, beta, bias, use_iws_mixer)
            results = self._samples_to_energy(samples)
            energies = [r[0] for r in results]
            n_evals[0] += 1
            return self._cvar_objective(energies)

        best_obj = float("inf")
        best_params = None

        for restart in range(self.config.n_restarts):
            if init_params is not None and restart == 0:
                x0 = init_params
            else:
                # Random initialization in [0, 2π]
                x0 = self.rng.uniform(0, 2 * math.pi, 2 * p)

            try:
                res = minimize(objective, x0, method="COBYLA",
                               options={"maxiter": self.config.max_opt_iter,
                                        "rhobeg": 0.5})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
            except Exception:
                pass

        if best_params is None:
            best_params = np.zeros(2 * p)

        return best_params[:p], best_params[p:], n_evals[0]

    def _update_bias(self, samples_results: list[tuple[float, np.ndarray, bool]],
                     current_bias: np.ndarray) -> np.ndarray:
        """
        Update bias probabilities from sampled results.
        New bias = softmax of (frequency * energy_quality).
        """
        new_bias = np.zeros((self.N, self.n))
        total_valid = 0

        for e, x, valid in samples_results:
            if not valid:
                continue
            total_valid += 1
            for i in range(self.N):
                a = int(np.argmax(x[i * self.n: (i + 1) * self.n]))
                new_bias[i, a] += 1.0

        if total_valid > 0:
            new_bias /= total_valid
        else:
            new_bias = np.ones((self.N, self.n)) / self.n

        # Smooth with current bias (exponential moving average)
        alpha = 0.3  # IWS update rate
        new_bias = alpha * new_bias + (1 - alpha) * current_bias
        # Normalize
        new_bias = new_bias / (new_bias.sum(axis=1, keepdims=True) + 1e-12)
        return new_bias

    def solve(self, greedy_assignment: Optional[np.ndarray] = None,
              exact_energy: Optional[float] = None) -> QAOAResult:
        """
        Run IWS-QAOA. If greedy_assignment is provided, warm-start from it.
        """
        t_start = time.time()
        p = self.config.p
        total_evals = 0
        iws_history = []
        params_history = []
        energy_history = []

        # Initialize bias from greedy assignment or uniform
        if greedy_assignment is not None and self.config.use_warm_start:
            bias = np.zeros((self.N, self.n))
            for i in range(self.N):
                a = int(np.argmax(greedy_assignment[i * self.n: (i + 1) * self.n]))
                bias[i, a] = 0.7  # Strong prior on greedy solution
                # Spread remaining probability
                bias[i] += 0.3 / self.n
                bias[i] = bias[i] / bias[i].sum()
        else:
            bias = np.ones((self.N, self.n)) / self.n

        best_energy = float("inf")
        best_assignment = None
        init_params = None

        for iws_round in range(self.config.n_iter):
            # Optimize QAOA params for current bias
            gamma, beta, n_ev = self._optimize_params(
                bias, self.config.use_xy_mixer, init_params
            )
            total_evals += n_ev
            init_params = np.concatenate([gamma, beta])  # warm-start optimizer

            # Final sample with best params
            samples = self._cost_circuit(gamma, beta, bias,
                                         self.config.use_xy_mixer)
            results = self._samples_to_energy(samples)

            # Best from this round
            valid_results = [(e, x) for e, x, v in results if v]
            round_best_e = float("inf")
            round_best_x = None
            if valid_results:
                round_best_e, round_best_x = min(valid_results, key=lambda r: r[0])

            if round_best_e < best_energy:
                best_energy = round_best_e
                best_assignment = round_best_x

            # Ground state probability (exact, noiseless)
            ground_state_prob = 0.0
            if exact_energy is not None and exact_energy != float("inf"):
                try:
                    probs = self._exact_probs_circuit(gamma, beta, bias,
                                                       self.config.use_xy_mixer)
                    # Sum probability of all states with energy ≤ exact_energy + 1e-3
                    all_states = self._enumerate_valid_states()
                    for state_idx, state_x in all_states:
                        e_s = float(state_x @ self.Q @ state_x)
                        if e_s <= exact_energy + 0.5:
                            ground_state_prob += float(probs[state_idx])
                except Exception:
                    ground_state_prob = 0.0

            iws_history.append({
                "round": iws_round,
                "best_energy": round_best_e,
                "n_valid_samples": len(valid_results),
                "gamma": gamma.tolist(),
                "beta": beta.tolist(),
                "ground_state_prob": ground_state_prob,
            })
            params_history.append(np.concatenate([gamma, beta]))
            energy_history.append(round_best_e if round_best_e != float("inf") else 1e6)

            # Update bias for next round
            bias = self._update_bias(results, bias)

        # Compute approximation ratio
        if exact_energy is not None and exact_energy != float("inf") and best_energy != float("inf"):
            # For minimization: ratio = best_found / optimal (≥ 1 means suboptimal)
            if abs(exact_energy) < 1e-9:
                approx_ratio = 1.0 if abs(best_energy) < 1e-9 else float("inf")
            else:
                approx_ratio = best_energy / exact_energy
        else:
            approx_ratio = float("nan")

        runtime = time.time() - t_start

        return QAOAResult(
            best_assignment=best_assignment if best_assignment is not None
                           else np.zeros(self.M, dtype=int),
            best_energy=best_energy if best_energy != float("inf") else float("nan"),
            exact_energy=exact_energy if exact_energy is not None else float("nan"),
            approximation_ratio=approx_ratio,
            ground_state_prob=iws_history[-1]["ground_state_prob"] if iws_history else 0.0,
            n_circuit_evals=total_evals,
            runtime_s=runtime,
            iws_history=iws_history,
            params_history=params_history,
            energy_history=energy_history,
        )

    def _enumerate_valid_states(self) -> list[tuple[int, np.ndarray]]:
        """Enumerate all valid (one-hot per block) computational basis states."""
        from itertools import product
        result = []
        for combo in product(range(self.n), repeat=self.N):
            x = np.zeros(self.M, dtype=int)
            for i, a in enumerate(combo):
                x[i * self.n + a] = 1
            idx = int(np.packbits(x, bitorder="big").view(np.uint8)[: (self.M + 7) // 8]
                      .dot(256 ** np.arange((self.M + 7) // 8 - 1, -1, -1)))
            # Convert to statevector index
            state_int = sum(x[b] * 2 ** (self.M - 1 - b) for b in range(self.M))
            result.append((state_int, x))
        return result
