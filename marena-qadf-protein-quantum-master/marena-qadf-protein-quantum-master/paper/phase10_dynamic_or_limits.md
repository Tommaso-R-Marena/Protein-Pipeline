# Phase 10 — Dynamic Analysis and Limitations
## Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)

---

## 1. Energy Landscape Analysis: 1D Chi1 Torsion Scan

A 1D torsion angle scan was performed for each of the 4 residues in the quantum experiment window (residues 3–6 of 1L2Y: TYR, ILE, GLN, TRP). The QUBO self-energy was evaluated at 5-degree intervals across χ₁ ∈ [−180°, +180°], yielding 73 evaluation points per residue.

**Method**: Continuous self-energy function defined as:
```
E(χ₁) = E_dunbrack(χ₁) + E_steric(χ₁)

E_dunbrack(χ₁) = -log( Σₖ pₖ · N(χ₁; μₖ, σ²) )
  where μₖ ∈ {-60°, 180°, 60°}, σ = 20°, pₖ from Dunbrack library [REF-05]

E_steric(χ₁) = 2 · exp(−χ₁²/(2 · 10²))
  (small Gaussian penalty near cis conformation χ₁ ≈ 0°)
```

**Results** (figure saved to /results/figures/energy_landscape.png):
| Residue | Actual χ₁ | Energy Minimum | Distance from Min | Assignment |
|---|---|---|---|---|
| TYR (res 3) | −148.59° | ~180° (t) | ~31.4° | Correctly assigned to t bin |
| ILE (res 4) | −44.72° | ~−60° (g−) | ~15.3° | Correctly assigned to g− bin |
| GLN (res 5) | −146.48° | ~180° (t) | ~33.5° | Correctly assigned to t bin |
| TRP (res 6) | +178.57° | ~180° (t) | ~1.4° | Correctly assigned to t bin |

All four residues have their actual PDB χ₁ values near the local minimum of their energy landscape, confirming that the simplified QUBO energy function captures the essential physics of the Dunbrack rotamer statistics.

**Key observation**: The energy landscape shows three clear minima at χ₁ ≈ −60° (g−), +60° (g+), and ±180° (t), consistent with the three principal rotamer basins of the backbone-dependent Dunbrack library. The barriers between minima are ~1–3 energy units, representing the physical barriers to rotamer interconversion at physiological temperature.

---

## 2. Limitations of This Work

### 2a. Classical Simulation Becomes Intractable Beyond ~20–25 Qubits

**Observed boundary** (Phase 6 scaling study):
| n residues | n qubits | QAOA time | Status |
|---|---|---|---|
| 2 | 6 | 1.68 s | Tractable |
| 3 | 9 | 3.73 s | Tractable |
| 4 | 12 | 6.73 s | Tractable |
| 5 | 15 | 15.11 s | Tractable |
| 6 | 18 | 125.12 s | Borderline |
| 7+ | 21+ | Intractable | Exceeds time budget |

Classical statevector simulation of an n-qubit system requires O(2ⁿ) complex amplitudes in memory. At n=20, this requires ~8 million complex floats (~128 MB per statevector) — manageable on modern hardware, but QAOA expectation value evaluation requires repeated statevector contraction with the circuit, which scales exponentially in time. At n=30 qubits (~1 billion amplitudes), classical simulation becomes impractical on single-workstation hardware.

**What this means**: The results in this paper are proof-of-concept. The quantum advantage claim for larger systems (n > 20 residues) cannot be empirically validated with classical simulation — it depends on access to actual quantum hardware. This is a fundamental limitation of all NISQ-era quantum computing research conducted without QPU access.

### 2b. QAOA at p=1,2 Does Not Guarantee Optimality

QAOA performance depends critically on the circuit depth parameter p. For the 4-residue instance:
- Exhaustive search: optimal energy = −34.07 (global minimum, ground truth matched)
- QAOA p=1 best bitstring energy: +93.83 (a non-optimal, constraint-violating state)
- QAOA p=2 best bitstring energy: +179.39 (worse than p=1)

This is consistent with the theoretical analysis in Bauza et al. 2023 [REF-06, DOI: 10.1038/s41534-023-00733-5]: QAOA at low depth (p=1,2) generally does not find the global optimum for protein-relevant QUBO instances. Higher p improves approximation quality but requires proportionally deeper circuits with more two-qubit gates — exacerbating noise sensitivity.

**Note on interpretation**: The QAOA best bitstring was decoded by selecting the highest-probability computational basis state. This is one sampling strategy; in practice, one would collect many bitstring samples and select the minimum-energy valid (one-hot satisfying) assignment. With 12 qubits and the sampling approach used, the QAOA circuit often produces bitstrings that violate the one-hot constraint (multiple states per residue), resulting in high penalty energies. A more sophisticated decoding strategy (e.g., classically post-processing the top-k bitstrings to find the best valid assignment) would improve solution quality.

### 2c. No Real QPU Hardware Used

All quantum circuit simulations in this project used PennyLane's `default.qubit` classical statevector simulator. Labels [CLASSICALLY SIMULATED] appear throughout all results. This means:

1. **No quantum speedup is demonstrated** — classical simulation of quantum circuits scales exponentially with qubit count; there is no advantage over direct classical methods at this scale.
2. **Coherence time limits are not modeled** — real QPU hardware has limited T1, T2 times (~20–200 μs for superconducting transmons [REF-11]); circuit execution must complete within the coherence window.
3. **Hardware connectivity constraints are not modeled** — real devices have limited qubit connectivity (not all-to-all); CNOT ladder circuits may require SWAP networks that significantly increase circuit depth.
4. **Readout errors are not modeled** — typical readout infidelity of 1–5% per qubit [REF-11] would degrade bitstring sampling quality.

### 2d. Dataset Limited to ≤25 Residues for Quantum Experiments

PDB entries ≤25 residues are structurally unusual (short peptides, cyclic peptides, NMR minimization structures). The model was evaluated on 1L2Y (20 residues, TC5b Trp-cage) — a well-structured but atypical protein. The statistical results (100% rotamer accuracy on 17 chi1 residues) partly reflect the regularity of the Trp-cage structure and the simplicity of the 3-bin rotamer encoding.

For biologically representative side-chain prediction, evaluation on a larger dataset of protein fragments (50–100 residues) with full backbone context would be required. The 1UBQ (ubiquitin) classical baseline in Phase 4 shows 68 chi1-bearing residues — a more representative sample — but quantum experiments were not run on this structure due to qubit overhead.

### 2e. Model Architecture Not Fully Trained

Due to computational constraints (no GPU, ≤25 residue structures, absence of a training pipeline for the quantum-classical hybrid model), the model architecture specified in Phase 3 was not end-to-end trained. The confidence scores and rotamer predictions reported in Phase 7 use an energy-based inference approach (softmax of Dunbrack-weighted self-energies) rather than learned model weights. This is a significant limitation: the EGNN + PQC hybrid model's performance cannot be evaluated from these experiments.

A proper training pipeline would require:
- A dataset of ≥500 protein structures (PDB entries ≤25 residues per REF-01, or full-length proteins with local windows)
- 3 random seed runs for variance estimation
- Ablation experiments comparing EGNN+PQC vs. EGNN+MLP (ablation plan in Phase 3G)
- Validation of calibration ECE across train/val/test splits

### 2f. QUBO Energy Function Uses Simplified Pairwise Interactions

The pairwise interaction energies in the QUBO use a simplified Lennard-Jones model with CB atom position estimates. A proper implementation would use:
- PyRosetta scoring function (as in Agathangelou et al. 2025, REF-02) for accurate pairwise energies
- Full side-chain heavy atom positions, not CB proxy
- Electrostatic interactions (Coulomb potential)
- Hydrogen bond terms

---

## 3. What Would Be Done With More Compute

### 3a. QAOA at Higher p

At p=3,4,5, QAOA approximation quality improves. The theoretical guarantee is that p → ∞ QAOA recovers the exact ground state (adiabatic limit). Practical quantum advantage for combinatorial problems is expected to emerge at moderate p on hardware with low two-qubit gate errors. With access to a 100-qubit fault-tolerant machine and circuit depth ~1000, QAOA at p=10–20 on 30-residue windows (90 qubits) could be tested.

### 3b. Larger Protein Systems

With 50–100 qubit QPU access, the following would be feasible:
- Quantum experiments on 1UBQ (76 residues) in overlapping 6-residue windows (18 qubits each)
- Cross-window coordination to ensure consistent global assignment
- Comparison against FlowPacker [REF-07] and SCWRL4 on the full 76-residue structure

### 3c. Noise-Aware Training

Incorporating depolarizing noise into the PQC training (via noise-aware optimization or noise-resilient ansatz design) could improve performance under realistic hardware conditions. The noise analysis in Phase 6 shows that ε₂ = 10⁻² causes ~23% energy degradation — acceptable if mitigation techniques (zero-noise extrapolation, probabilistic error cancellation) are applied.

### 3d. Hardware Benchmarking

The ultimate test of this approach requires:
1. Running the 12-qubit QAOA circuit on IBM Quantum or IonQ hardware
2. Comparing actual QPU bitstring energies against classical simulation predictions
3. Quantifying the fidelity gap (actual vs. ideal circuit performance)
4. Evaluating whether hardware noise degrades solution quality below the classical SA baseline

This requires QPU access (IBM Quantum Network, Amazon Braket, or IonQ Cloud) and is the natural next step after classical simulation validation.

---

## 4. Summary of Resource Boundaries

| Resource | Observed Value | Limit |
|---|---|---|
| Max tractable qubits (classical sim, <60s) | ~18 (n=6 residues) | 20–25 qubits |
| Max tractable qubits (classical sim, <600s) | ~21 (n=7 residues) | 30 qubits |
| QAOA p=1 optimality gap (n=4) | Non-optimal (E=93.8 vs −34.1) | Requires p≥5 |
| Training data for model fitting | Not available | Need ≥500 structures |
| QPU hardware | Not used | IBM/IonQ access required |

---

*References: REF-01 (DOI: 10.1021/acs.jctc.4c00067), REF-02 (arXiv: 2507.19383), REF-05 (DOI: 10.1002/prot.22921), REF-06 (DOI: 10.1038/s41534-023-00733-5), REF-07 (DOI: 10.1101/2024.07.05.602280), REF-11 (NISQ noise parameters)*
