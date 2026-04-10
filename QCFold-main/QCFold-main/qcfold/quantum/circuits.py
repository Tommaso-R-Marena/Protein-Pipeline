"""
Quantum circuits for fold-state optimization using PennyLane.

Implements:
  1. QAOA ansatz for QUBO optimization of fold-state assignment
  2. Hardware-efficient variational ansatz (VQE-style)
  3. Measurement and decoding of fold-state assignments

The quantum circuit encodes the fold-state assignment as qubit states:
  |0⟩ = Fold A, |1⟩ = Fold B for each residue.

The QAOA circuit alternates between:
  - Cost layer: encodes the QUBO Hamiltonian (Z-Z interactions + Z biases)
  - Mixer layer: enables state transitions (X rotations)

The VQE circuit uses a hardware-efficient ansatz with:
  - Single-qubit rotations (RY, RZ)
  - CNOT entangling layers
  - Parameterized depth for expressivity control
"""

import numpy as np
from typing import Tuple, Dict, Optional, List

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

from .qubo import QUBOInstance


class QAOAFoldOptimizer:
    """QAOA-based optimizer for fold-state assignment.
    
    Maps the fold-switching QUBO to a quantum circuit and optimizes
    the variational parameters to find the lowest-energy assignment.
    """

    def __init__(
        self,
        qubo: QUBOInstance,
        num_layers: int = 4,
        backend: str = "default.qubit",
        shots: Optional[int] = None,
        seed: int = 42,
    ):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for quantum circuits")

        self.qubo = qubo
        self.n_qubits = qubo.n_residues
        self.num_layers = num_layers
        self.backend = backend
        self.shots = shots
        self.seed = seed

        # Convert QUBO to Ising coefficients
        self.h, self.J, self.offset = qubo.to_ising()

        # Build the PennyLane device and circuit
        self.dev = qml.device(backend, wires=self.n_qubits, shots=shots)
        self._build_cost_hamiltonian()
        self._build_circuit()

    def _build_cost_hamiltonian(self):
        """Build the cost Hamiltonian as a PennyLane Observable."""
        coeffs = []
        obs = []

        # Single-qubit Z terms
        for i in range(self.n_qubits):
            if abs(self.h[i]) > 1e-10:
                coeffs.append(self.h[i])
                obs.append(qml.PauliZ(i))

        # Two-qubit ZZ terms
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(self.J[i, j]) > 1e-10:
                    coeffs.append(self.J[i, j])
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

        if not coeffs:
            # Trivial Hamiltonian
            coeffs = [0.0]
            obs = [qml.Identity(0)]

        self.cost_hamiltonian = qml.Hamiltonian(coeffs, obs)

    def _build_circuit(self):
        """Build the QAOA circuit as a PennyLane QNode."""
        cost_h = self.cost_hamiltonian
        n_qubits = self.n_qubits
        n_layers = self.num_layers

        @qml.qnode(self.dev)
        def qaoa_circuit(params):
            gammas = params[:n_layers]
            betas = params[n_layers:]

            # Initial superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            # QAOA layers
            for layer in range(n_layers):
                # Cost layer: exp(-i * gamma * H_C)
                qml.ApproxTimeEvolution(cost_h, gammas[layer], 1)

                # Mixer layer: exp(-i * beta * H_M) where H_M = sum X_i
                for i in range(n_qubits):
                    qml.RX(2 * betas[layer], wires=i)

            return qml.expval(cost_h)

        self.circuit = qaoa_circuit
        self.num_params = 2 * n_layers

    def optimize(
        self,
        max_iterations: int = 200,
        lr: float = 0.01,
        convergence_threshold: float = 1e-6,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run QAOA optimization.
        
        Returns:
            best_assignment: (n_residues,) binary array
            best_energy: float
            history: dict with optimization metrics
        """
        np.random.seed(self.seed)
        params = pnp.array(
            np.random.uniform(0, np.pi, self.num_params),
            requires_grad=True,
        )

        optimizer = qml.AdamOptimizer(stepsize=lr)
        history = {"energies": [], "params": []}
        best_energy = float("inf")
        best_params = params.copy()

        for step in range(max_iterations):
            params, energy = optimizer.step_and_cost(self.circuit, params)
            energy_val = float(energy)
            history["energies"].append(energy_val)

            if energy_val < best_energy:
                best_energy = energy_val
                best_params = params.copy()

            if verbose and step % 20 == 0:
                print(f"  QAOA step {step}: energy = {energy_val:.6f}")

            # Convergence check
            if len(history["energies"]) > 10:
                recent = history["energies"][-10:]
                if max(recent) - min(recent) < convergence_threshold:
                    if verbose:
                        print(f"  Converged at step {step}")
                    break

        # Sample the optimized circuit to get the assignment
        best_assignment = self._sample_assignment(best_params)
        qubo_energy = self.qubo.evaluate(best_assignment)

        history["final_params"] = best_params
        history["num_steps"] = step + 1
        history["qubo_energy"] = qubo_energy

        return best_assignment, qubo_energy, history

    def _sample_assignment(
        self, params: np.ndarray, num_samples: int = 100
    ) -> np.ndarray:
        """Sample bit strings from the optimized circuit and return the best."""
        n_qubits = self.n_qubits
        n_layers = self.num_layers
        cost_h = self.cost_hamiltonian

        # Create a sampling circuit
        dev_sample = qml.device(
            self.backend, wires=n_qubits, shots=num_samples
        )

        @qml.qnode(dev_sample)
        def sample_circuit(params):
            gammas = params[:n_layers]
            betas = params[n_layers:]
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(n_layers):
                qml.ApproxTimeEvolution(cost_h, gammas[layer], 1)
                for i in range(n_qubits):
                    qml.RX(2 * betas[layer], wires=i)
            return qml.sample()

        samples = sample_circuit(params)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Evaluate each sample and return the best
        best_energy = float("inf")
        best_sample = samples[0]
        for s in samples:
            e = self.qubo.evaluate(s)
            if e < best_energy:
                best_energy = e
                best_sample = s

        return np.array(best_sample, dtype=int)


class VQEFoldOptimizer:
    """VQE with hardware-efficient ansatz for fold-state assignment.
    
    Uses RY-RZ single-qubit rotations with CNOT entangling layers.
    More flexible than QAOA for capturing complex correlations.
    """

    def __init__(
        self,
        qubo: QUBOInstance,
        circuit_depth: int = 6,
        backend: str = "default.qubit",
        seed: int = 42,
    ):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required")

        self.qubo = qubo
        self.n_qubits = qubo.n_residues
        self.circuit_depth = circuit_depth
        self.seed = seed

        self.h, self.J, self.offset = qubo.to_ising()

        self.dev = qml.device(backend, wires=self.n_qubits)
        self._build_cost_hamiltonian()
        self._build_circuit()

    def _build_cost_hamiltonian(self):
        """Build cost Hamiltonian."""
        coeffs, obs = [], []
        for i in range(self.n_qubits):
            if abs(self.h[i]) > 1e-10:
                coeffs.append(self.h[i])
                obs.append(qml.PauliZ(i))
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if abs(self.J[i, j]) > 1e-10:
                    coeffs.append(self.J[i, j])
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:
            coeffs, obs = [0.0], [qml.Identity(0)]
        self.cost_hamiltonian = qml.Hamiltonian(coeffs, obs)

    def _build_circuit(self):
        """Build VQE circuit with hardware-efficient ansatz."""
        n_q = self.n_qubits
        depth = self.circuit_depth
        cost_h = self.cost_hamiltonian

        @qml.qnode(self.dev)
        def vqe_circuit(params):
            # params shape: (depth, n_qubits, 2) for RY and RZ
            for d in range(depth):
                for i in range(n_q):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                # CNOT entangling layer (linear connectivity)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Circular entanglement for last qubit
                if n_q > 2:
                    qml.CNOT(wires=[n_q - 1, 0])
            return qml.expval(cost_h)

        self.circuit = vqe_circuit
        self.param_shape = (depth, n_q, 2)

    def optimize(
        self,
        max_iterations: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run VQE optimization."""
        np.random.seed(self.seed)
        params = pnp.array(
            np.random.uniform(0, 2 * np.pi, self.param_shape),
            requires_grad=True,
        )

        optimizer = qml.AdamOptimizer(stepsize=lr)
        history = {"energies": []}
        best_energy = float("inf")
        best_params = params.copy()

        for step in range(max_iterations):
            params, energy = optimizer.step_and_cost(self.circuit, params)
            energy_val = float(energy)
            history["energies"].append(energy_val)

            if energy_val < best_energy:
                best_energy = energy_val
                best_params = params.copy()

            if verbose and step % 20 == 0:
                print(f"  VQE step {step}: energy = {energy_val:.6f}")

        # Sample assignment
        best_assignment = self._sample_assignment(best_params)
        qubo_energy = self.qubo.evaluate(best_assignment)

        history["final_params"] = best_params
        history["qubo_energy"] = qubo_energy

        return best_assignment, qubo_energy, history

    def _sample_assignment(self, params, num_samples=100):
        """Sample from optimized VQE circuit."""
        n_q = self.n_qubits
        depth = self.circuit_depth

        dev_sample = qml.device(self.backend if hasattr(self, 'backend')
                                else "default.qubit",
                                wires=n_q, shots=num_samples)

        @qml.qnode(dev_sample)
        def sample_circuit(params):
            for d in range(depth):
                for i in range(n_q):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_q > 2:
                    qml.CNOT(wires=[n_q - 1, 0])
            return qml.sample()

        samples = sample_circuit(params)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        best_e = float("inf")
        best_s = samples[0]
        for s in samples:
            e = self.qubo.evaluate(s)
            if e < best_e:
                best_e = e
                best_s = s
        return np.array(best_s, dtype=int)
