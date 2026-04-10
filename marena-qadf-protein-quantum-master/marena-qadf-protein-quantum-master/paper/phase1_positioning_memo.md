# Phase 1 — Scientific Positioning Memo
## Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)
### Target Subproblem: Side-Chain Rotamer Optimization

---

## 1. Problem Definition

Given a **fixed protein backbone** (set of Cα, N, C, O coordinates), the side-chain rotamer optimization problem asks: *What is the minimum-energy assignment of discrete rotamer states (χ₁, χ₂, … torsion angles) for each residue simultaneously?*

### Formal statement

Let R = {r₁, r₂, …, rₙ} be n residues with fixed backbone, and let each residue rᵢ have a finite set Sᵢ of feasible rotamer states drawn from the backbone-dependent Dunbrack rotamer library [REF-05, DOI: 10.1002/prot.22921]. The combinatorial energy minimization problem is:

```
minimize  E(s₁, s₂, …, sₙ) = Σᵢ Eself(rᵢ, sᵢ) + Σᵢ<ⱼ Epair(rᵢ, sᵢ, rⱼ, sⱼ)
subject to sᵢ ∈ Sᵢ  ∀i
```

where **Eself** is the self-energy of rotamer sᵢ (steric clashes with backbone, Dunbrack log-probability penalty) and **Epair** is the pairwise interaction energy (van der Waals, electrostatics, hydrogen bonding between residue side chains).

The search space grows as O(|S|ⁿ). For |S| = 3 discrete bins per residue: 3¹⁰ ≈ 59,000 states at 10 residues; 3²⁰ ≈ 3.5 × 10⁹ at 20 residues. This combinatorial explosion makes even moderate-length peptides hard for exhaustive classical search.

---

## 2. Why Side-Chain Rotamer Optimization Is the Right Subproblem

### 2a. Literature Justification

**Doga et al. 2024 (JCTC, REF-01, DOI: 10.1021/acs.jctc.4c00067)** introduce the Quantum Advantage Decision Framework (QADF) for systematically ranking protein structure prediction subproblems on their suitability for quantum hardware. Their key criteria are: (i) biological importance, (ii) search space discreteness, (iii) QUBO/Ising Hamiltonian compatibility, (iv) qubit overhead, and (v) NISQ-era circuit depth. Side-chain packing scores favorably on all five: the Dunbrack library discretizes chi angles into a finite rotamer alphabet, pairwise energies map directly to quadratic interaction terms in an Ising Hamiltonian, and a local window of 4–8 residues requires only 12–24 qubits — within NISQ reach.

**Agathangelou et al. 2025 (arXiv: 2507.19383, REF-02)** provide the most direct published precedent for QUBO-encoded side-chain optimization. They formulate rotamer selection as an Ising Hamiltonian, implement QAOA with a local XY mixer in Qiskit, and compare against dual annealing (L-BFGS-B). Their analysis demonstrates that (a) the problem structure is naturally suited to QUBO, (b) COBYLA optimization with CVaR (α=0.2) improves convergence, and (c) a crossing point exists where quantum methods may outperform classical scaling.

This project extends that foundation by adding: (1) a calibrated confidence estimation head (absent from Agathangelou et al.), (2) an EGNN classical backbone trained end-to-end (not a pure energy-minimization pipeline), and (3) a systematic evaluation on real PDB data with statistical rigor (bootstrap CIs, Wilcoxon tests).

### 2b. Why This Subproblem — Not Others

| Subproblem | Fixed backbone? | Discrete state space? | Qubit estimate (n=5) | Near-term feasible? |
|---|---|---|---|---|
| Side-chain rotamer optimization | **YES** | **YES** (Dunbrack bins) | **15 qubits** | **YES** |
| Catalytic loop refinement | YES (rest of protein) | Partially | ~20 qubits | Marginal |
| Short peptide conformation search | NO | Partially (lattice) | ~14 qubits | Marginal |
| Global backbone folding | NO | NO (continuous) | >100 qubits | NO |

The fixed-backbone assumption reduces problem dimensionality dramatically and is biologically valid: backbone coordinates are increasingly well-determined by (a) experimental structures deposited in the PDB, (b) homology modeling, or (c) AlphaFold 2 global fold prediction. Once backbone is known, the remaining uncertainty concentrates in side-chain placement — precisely where rotamer optimization applies.

---

## 3. The Specific Novelty Gap

Existing work leaves three gaps that this project addresses:

### Gap 1: No QADF + Calibrated Confidence Integration

Doga et al. (REF-01) introduce subproblem selection criteria but do not implement a model with a calibrated per-residue confidence output analogous to AlphaFold's pLDDT. Agathangelou et al. (REF-02) compare solution quality (energy) but provide no confidence metric for individual residue predictions.

This project is — to our knowledge — the **first to combine QADF-guided subproblem selection with a calibrated confidence head** for side-chain rotamer prediction, producing a pLDDT-style score with validated calibration (ECE < target threshold).

### Gap 2: pLDDT Is Not a Calibrated Probability

CalPro (REF-08, arXiv: 2601.07201) demonstrates that AlphaFold 2's pLDDT exhibits systematic miscalibration under distribution shift, with 15–25% coverage degradation for baseline methods. CalPro's evidential-conformal framework reduces calibration error by 30–50%. This motivates the explicit calibration regularizer in our loss function (Phase 3).

### Gap 3: No Working Prototype on Real PDB Data at Classical Simulation Scale

Prior QUBO formulations for rotamer optimization use synthetic or model systems. This project downloads real PDB structures (1L2Y, 1UBQ), extracts actual chi angles, constructs QUBO matrices from true pairwise distances, and runs all quantum experiments as full classical simulations of the quantum circuit — providing a reproducible baseline at the scale available to any research group without QPU access.

---

## 4. AlphaFold Gap Analysis

### 4a. What AlphaFold 2 Reports for Side-Chains

AlphaFold 2 (Jumper et al. 2021, Nature 596:583–589, DOI: 10.1038/s41586-021-03819-2) revolutionized backbone prediction. However:

- **χ₁/χ₂ side-chain recovery rates are NOT reported** in the primary CASP14 evaluation paper (Jumper et al. 2021, Proteins, DOI: 10.1002/prot.26257, REF-04). The paper is focused exclusively on backbone GDT_TS and TM-score metrics.
- pLDDT regresses true per-residue lDDT-Cα — a **backbone** metric — not a side-chain accuracy metric.
- AlphaFold 2 does not produce a per-residue confidence score for chi angle accuracy.

### 4b. pLDDT Miscalibration

The pLDDT score is commonly interpreted as a probability of correctness. CalPro (REF-08) shows this is incorrect: pLDDT is **not a calibrated probability** and exhibits 15–25% coverage degradation under distribution shift. This is a fundamental limitation for downstream tasks (e.g., docking, drug design) that rely on residue-level confidence.

### 4c. Low-pLDDT Disordered Regions

AlphaFold assigns low pLDDT (<50) to intrinsically disordered regions (IDRs). AlphaFold-Metainference (Nature Communications 2025, DOI: 10.1038/s41467-025-56572-9, REF-14) explicitly acknowledges that long low-pLDDT regions indicate genuine structural disorder not predictable by single-structure AlphaFold. These regions represent an unsolved subproblem where a quantum-hybrid rotamer optimizer — operating on local windows — could provide value by sampling the rotamer ensemble rather than committing to a single prediction.

### 4d. Failure Case: T1047s1-D1

In the CASP14 evaluation (REF-04, Table 1B), AlphaFold 2's best-of-5 models achieved GDT_TS = **50.47** for domain T1047s1-D1 (a large beta-sheet target with a β-strand packed at the wrong angle). This demonstrates that even state-of-the-art backbone predictors fail on specific structural topologies — reinforcing that backbone and side-chain prediction remain open problems.

---

## 5. Dataset Selection

### 5a. Quantum Experiments: PDB Structures ≤25 Residues

**Primary dataset**: PDB entry **1L2Y** — the TC5b Trp-cage mini-protein (20 residues, 2 chains resolved, NMR). Selected because:
- 20 residues → 60 binary variables for 3-state rotamer encoding → 60-qubit theoretical upper bound
- Local windows of 4–6 residues require 12–18 qubits (NISQ-feasible with classical simulation)
- Well-characterized structure; frequently used in computational benchmarks
- Available via RCSB PDB API (REF-10)

Exclusion criteria for quantum experiments:
- Proteins >25 residues (too many qubits for meaningful simulation within time budget)
- Structures with >10% missing side-chain atoms
- Non-standard amino acids without Dunbrack rotamer entries

### 5b. Classical Baseline: ~50-Residue System

**Secondary dataset**: PDB entry **1UBQ** — ubiquitin (76 residues, 1 chain). Selected because:
- 76 residues is representative of typical CASP targets
- Ubiquitin is a benchmark protein used extensively in prior computational work
- Too large for direct quantum simulation → used for classical baseline (greedy, SA) only
- Allows comparison: does the quantum-assisted approach for small windows generalize to fragments of a larger protein?

### 5c. Train/Validation/Test Split

Following standard practice for protein machine learning:
- **70% train / 15% validation / 15% test**
- Split by **sequence identity clustering** at 30% identity threshold (using CD-HIT or MMseqs2) to prevent homology leakage
- For the 1L2Y NMR ensemble (20 models in PDB entry): model 1 → quantum experiments; models 2–5 → calibration validation; models 6–20 → held-out test
- All splits recorded in /data/rotamers/splits.json (produced in Phase 4)

---

## 6. Summary of Novel Contributions

| Contribution | Prior art | This work |
|---|---|---|
| QADF subproblem selection | Doga et al. 2024 (REF-01) | Applies QADF to side-chain rotamers explicitly |
| QUBO encoding of rotamer optimization | Agathangelou et al. 2025 (REF-02) | Extends with real PDB data, 1L2Y/1UBQ |
| Calibrated confidence per residue | CalPro 2026 (REF-08) | First integration with QADF/QAOA pipeline |
| Working prototype on real PDB data | Not in prior QAOA rotamer work | This project (Phases 4–7) |
| [CLASSICALLY SIMULATED] quantum results | Standard practice | Explicitly labeled throughout |

---

*References: REF-01 (DOI: 10.1021/acs.jctc.4c00067), REF-02 (arXiv: 2507.19383), REF-04 (DOI: 10.1002/prot.26257), REF-05 (DOI: 10.1002/prot.22921), REF-08 (arXiv: 2601.07201), REF-10 (RCSB PDB API), REF-14 (DOI: 10.1038/s41467-025-56572-9)*
