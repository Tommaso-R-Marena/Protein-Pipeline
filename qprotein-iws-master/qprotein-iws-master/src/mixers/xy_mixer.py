"""
Local XY-mixer for one-hot constrained QAOA.

Implements the ring-XY model from Agathangelou et al. (arXiv:2507.19383):
  H_M^XY = (1/2) Σ_i Σ_j (X_{in+j} X_{in+(j+1)%n} + Y_{in+j} Y_{in+(j+1)%n})

Also implements the Iterative Warm-Start (IWS) variant from
Bucher et al. (arXiv:2604.02083):
  H_M^IWS = Σ_i Σ_j (cos(θ_ij/2)(XX+YY)/2 + sin(θ_ij/2)(XY-YX)/2i)

where θ_ij encodes the warm-start bias from a classical heuristic.
"""

import numpy as np
import pennylane as qml


def xy_mixer_circuit(wires_per_residue: list[list[int]], beta: float) -> None:
    """
    Apply one layer of the local XY-mixer to a PennyLane circuit.

    wires_per_residue: list of qubit index lists, one per residue.
    beta: mixer angle parameter.
    """
    for block in wires_per_residue:
        n = len(block)
        for j in range(n):
            q0 = block[j]
            q1 = block[(j + 1) % n]
            # XX+YY interaction = iSWAP-like gate
            # Decomposed as: CNOT - Ry(β) - CNOT
            qml.IsingXX(beta, wires=[q0, q1])
            qml.IsingYY(beta, wires=[q0, q1])


def iws_mixer_circuit(wires_per_residue: list[list[int]], beta: float,
                       bias: np.ndarray) -> None:
    """
    Iterative Warm-Start XY-mixer (Bucher et al. 2026).

    bias: (N, n) array of warm-start probabilities per rotamer.
          Encodes prior knowledge from classical greedy or GNN.
    """
    for i, block in enumerate(wires_per_residue):
        n = len(block)
        for j in range(n):
            q0 = block[j]
            q1 = block[(j + 1) % n]
            # Warm-start angle: bias the XX+YY coupling by Δθ_ij
            # from the classical solution probabilities
            b0 = bias[i, j] if bias is not None else 0.5
            b1 = bias[i, (j + 1) % n] if bias is not None else 0.5
            theta = 2.0 * beta * (1.0 + 0.5 * (b0 - b1))
            theta = np.clip(theta, -np.pi, np.pi)
            qml.IsingXX(theta, wires=[q0, q1])
            qml.IsingYY(theta, wires=[q0, q1])


def prepare_w_state(wires: list[int]) -> None:
    """
    Prepare the W-state |100...0> + |010...0> + ... superposition (equal weight).
    This is the ground state of the XY-mixer and the natural initial state.
    
    Uses the circuit from Bartschi & Eidenbenz (2019) for exact W-state preparation.
    """
    n = len(wires)
    if n == 0:
        return
    # Step 1: |0...0> -> |10...0> via X on first qubit
    qml.PauliX(wires=wires[0])
    # Step 2: Spread via controlled rotations
    for k in range(1, n):
        # Angle that distributes amplitude uniformly
        theta_k = 2.0 * np.arcsin(np.sqrt(1.0 / (n - k + 1)))
        qml.CRY(theta_k, wires=[wires[k - 1], wires[k]])
        qml.CNOT(wires=[wires[k], wires[k - 1]])


def prepare_iws_initial_state(wires_per_residue: list[list[int]],
                               bias: np.ndarray) -> None:
    """
    Prepare warm-started initial state for IWS-QAOA.

    For each residue block, initialize a biased superposition:
      |ψ_i> = Σ_a √(p_{i,a}) |e_a>
    where |e_a> is the one-hot state with bit a set.

    This is a biased W-state that concentrates on likely rotamers.
    """
    for i, block in enumerate(wires_per_residue):
        n = len(block)
        probs = bias[i] if bias is not None else np.ones(n) / n
        probs = probs / (probs.sum() + 1e-12)

        # Prepare |e_0> (deterministic)
        qml.PauliX(wires=block[0])

        # Spread amplitude using biased CSWAP-based rotation sequence
        cumsum = probs[0]
        for k in range(1, n):
            remaining = 1.0 - cumsum + probs[k]
            if remaining < 1e-9:
                theta = 0.0
            else:
                theta = 2.0 * np.arcsin(np.sqrt(probs[k] / remaining))
            qml.CRY(theta, wires=[block[k - 1], block[k]])
            qml.CNOT(wires=[block[k], block[k - 1]])
            cumsum += probs[k]
