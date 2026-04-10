# Phase 2 — QADF Framework: Subproblem Scoring and Classification
## Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)

---

## Overview

The Quantum Advantage Decision Framework (QADF), introduced by Doga et al. 2024 [REF-01, DOI: 10.1021/acs.jctc.4c00067], provides a systematic methodology for evaluating which protein structure prediction subproblems are amenable to near-term quantum advantage. We apply this framework to 8 candidate subproblems, scoring each on 9 dimensions.

### Scoring Criteria

| # | Criterion | Description | Scale |
|---|---|---|---|
| 1 | Biological importance | Impact on downstream structural biology tasks | 1–5 |
| 2 | Search space discreteness | Whether the problem admits a finite discrete state space | 1–5 |
| 3 | QUBO/Ising compatibility | Ease of mapping to quadratic unconstrained binary optimization | 1–5 |
| 4 | Estimated qubit count (n=5 residues) | Number of logical qubits required | (raw number) |
| 5 | Gate depth bottleneck | CNOT/entangling gate depth per QAOA layer | Low/Med/High |
| 6 | Noise sensitivity at NISQ error rates (ε₂ ~ 10⁻³–10⁻²) | Degradation under depolarizing noise [REF-11] | Low/Med/High |
| 7 | Classical baseline strength | How good is the best-known classical heuristic? | 1–5 (5=very strong) |
| 8 | Expected quantum benefit | Theoretical or empirical evidence for quantum speedup | 1–5 |
| 9 | Earliest plausible utility | Estimated timeline for meaningful quantum advantage | Near/Med/Far |

**Classification labels:**
- **A**: Near-term hybrid candidate (NISQ-feasible, classical simulation → hardware within 3–5 years)
- **B**: Medium-term target (requires fault-tolerant QC or significantly larger NISQ systems)
- **C**: Poor near-term target (continuous/high-dimensional space; fundamental obstacles to QUBO encoding)

---

## 2.1 Full Scoring Table

### Subproblem 1: Global Backbone Folding

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 5 | Core of protein structure prediction |
| Search space discreteness | 1 | Continuous φ/ψ/ω angles; no natural discretization |
| QUBO/Ising compatibility | 1 | Continuous energy landscape; poor QUBO fit |
| Estimated qubit count | >100 | Even 10-residue backbone: ≥100 qubits needed |
| Gate depth bottleneck | **High** | Exponentially many CNOT layers needed |
| Noise sensitivity | **High** | Long circuits; noise accumulates severely |
| Classical baseline strength | 5 | AlphaFold 2 near-solves CASP; GDT_TS median 92.4 [REF-04] |
| Expected quantum benefit | 1 | QAOA requires >40 layers for peptide folding [REF-06] |
| Earliest plausible utility | **Far** (>10 years) | |

**QADF Classification: C — Poor near-term target**

**Justification**: Bauza et al. 2023 [REF-06, DOI: 10.1038/s41534-023-00733-5] demonstrate conclusively that QAOA benchmarked on alanine tetrapeptide folding (Lennard-Jones, lattice model, up to 28 qubits [CLASSICALLY SIMULATED]) requires >40 ansatz layers to approach minimum energy, with performance matchable by random sampling with only 6.6× overhead. Their conclusion: *"These results cast serious doubt on the ability of QAOA to address the protein folding problem in the near term, even in an extremely simplified setting."* Furthermore, AlphaFold 2 achieves median domain GDT_TS of 92.4 on CASP14 [REF-04], leaving minimal room for quantum improvement on the backbone folding task itself.

---

### Subproblem 2: Short Peptide Conformational Search

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 3 | Relevant for cyclic peptide drugs, antimicrobials |
| Search space discreteness | 3 | Lattice models discretize; realistic models continuous |
| QUBO/Ising compatibility | 3 | HP lattice model maps to QUBO; realistic force fields do not |
| Estimated qubit count (n=5) | ~14 | Lattice encoding; O(n log n) qubits |
| Gate depth bottleneck | Medium | Moderate; lattice constraints tractable |
| Noise sensitivity | Medium | Mid-length circuits |
| Classical baseline strength | 3 | MD sampling competitive; lattice models solvable classically |
| Expected quantum benefit | 2 | Marginal; Grover speedup O(√N) not decisive |
| Earliest plausible utility | Medium (5–7 years) | |

**QADF Classification: B — Medium-term target**

**Notes**: The HP (hydrophobic-polar) lattice model maps cleanly to QUBO [REF-03, DOI: 10.1371/journal.pcbi.1011033], but is too simplified to be biologically relevant. Realistic peptide conformational search involves continuous dihedrals and empirical force fields — not QUBO-compatible in the near term.

---

### Subproblem 3: Catalytic Loop Refinement

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 5 | Catalytic loops determine enzyme function directly |
| Search space discreteness | 3 | Can discretize loop backbone to ~5–10 conformers |
| QUBO/Ising compatibility | 3 | Partial; pairwise backbone contacts → quadratic terms |
| Estimated qubit count (n=5) | ~20 | 5-residue loop: ~4 qubits/residue for backbone discretization |
| Gate depth bottleneck | Medium | |
| Noise sensitivity | Medium–High | |
| Classical baseline strength | 3 | RosettaLoops, kinematic closure are competitive |
| Expected quantum benefit | 3 | Doga et al. [REF-01] demonstrated P-loop prediction on 127-qubit IBM Eagle |
| Earliest plausible utility | Near–Medium (3–5 years) | |

**QADF Classification: B (border A)**

**Notes**: Doga et al. [REF-01] report proof-of-concept loop prediction on IBM's 127-qubit Eagle (R3) quantum processor using RealAmplitudes ansatz. This demonstrates near-term feasibility but requires real QPU access. For purely classical simulation scale, this subproblem is feasible but not the best choice given the larger qubit overhead compared to rotamer optimization.

---

### Subproblem 4: Side-Chain Packing / Rotamer Optimization ⭐ [PRIMARY]

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 5 | Determines binding affinity, enzyme catalysis, stability |
| Search space discreteness | **5** | Dunbrack library [REF-05] provides finite rotamer alphabet |
| QUBO/Ising compatibility | **5** | One-hot encoding → quadratic penalty terms [REF-02] |
| Estimated qubit count (n=5 residues, 3 states) | **15** | 3 binary vars × 5 residues = 15 qubits |
| Gate depth bottleneck | **Low** | QAOA p=1: 2 layers per edge; manageable circuit |
| Noise sensitivity | **Low–Medium** | Short circuits at p=1 tolerate ε₂ ~ 10⁻³ [REF-11] |
| Classical baseline strength | 4 | FlowPacker [REF-07] and SCWRL4 strong but not perfect |
| Expected quantum benefit | 4 | QUBO structure exploits native Ising hardware; crossing point shown [REF-02] |
| Earliest plausible utility | **Near (2–4 years)** | |

**QADF Classification: A — Near-term hybrid candidate** ✓

**Justification**: This subproblem scores highest overall due to the unique combination of:
1. **Natural discreteness**: Dunbrack rotamer library [REF-05, DOI: 10.1002/prot.22921] provides a backbone-dependent finite alphabet of rotamer states, with typically 3–9 states per amino acid type. This discretization is biologically justified and maps directly to binary variable encoding.
2. **QUBO compatibility**: Agathangelou et al. [REF-02, arXiv: 2507.19383] demonstrate exact QUBO formulation: H = Σᵢ hᵢ σᵢᶻ + Σᵢ<ⱼ Jᵢⱼ σᵢᶻ σⱼᶻ where hᵢ captures self-energies and Jᵢⱼ captures pairwise interaction energies from PyRosetta scoring.
3. **Low qubit overhead**: A 5-residue window with 3 rotamer states requires only 15 logical qubits — well within classical simulation and near NISQ hardware range.
4. **Fixed backbone assumption**: Backbone coordinates assumed fixed from PDB or AlphaFold prediction, reducing dimensionality by >80%.
5. **Biological relevance**: Side-chain placement determines protein-protein interfaces, drug binding pockets, and enzymatic activity — high-value downstream impact.

---

### Subproblem 5: Constrained Local Energy Minimization

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 3 | General energy refinement; less targeted than rotamers |
| Search space discreteness | 2 | Primarily continuous; grid quantization loses resolution |
| QUBO/Ising compatibility | 2 | Quadratic terms possible but continuous gradient flow preferred |
| Estimated qubit count | ~30+ | Fine grid discretization required |
| Gate depth bottleneck | High | Grover-based search needs high depth |
| Noise sensitivity | High | |
| Classical baseline strength | 5 | L-BFGS, gradient descent effectively solves this |
| Expected quantum benefit | 1 | Classical gradient descent dominates for continuous optimization |
| Earliest plausible utility | Far (>10 years) | |

**QADF Classification: C — Poor near-term target**

---

### Subproblem 6: Disulfide Bond Network Optimization

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 3 | Critical for proteins with multiple Cys; niche but impactful |
| Search space discreteness | **5** | Binary: Cys_i—Cys_j bonded or not bonded |
| QUBO/Ising compatibility | **5** | Perfect matching → min-weight matching → QUBO |
| Estimated qubit count (4 Cys) | ~6 | O(n²) binary variables for n Cys residues |
| Gate depth bottleneck | Low | |
| Noise sensitivity | Low | Very short circuits |
| Classical baseline strength | 4 | Bayesian optimization of disulfide bonds effective |
| Expected quantum benefit | 3 | Quadratic speedup plausible for large Cys networks |
| Earliest plausible utility | Near (2–4 years) | |

**QADF Classification: A (but niche)**

**Notes**: Excellent QUBO structure. However, most proteins have few Cys residues; the problem size is often too small to show quantum advantage, and classical algorithms solve small matching problems exactly. Classified A due to near-perfect QUBO mapping, but deprioritized due to limited biological scope.

---

### Subproblem 7: Protein-Protein Interface Packing

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 5 | PPI interfaces central to drug discovery, immunology |
| Search space discreteness | 4 | Interface residues form discrete rotamer problem |
| QUBO/Ising compatibility | 4 | Same one-hot rotamer encoding as sub-problem 4 |
| Estimated qubit count (10 interface residues) | ~30 | 3 states × 10 residues = 30 qubits |
| Gate depth bottleneck | Medium | More inter-residue interactions than single-chain |
| Noise sensitivity | Medium | |
| Classical baseline strength | 4 | Rosetta FastRelax; InterfaceAnalyzer competitive |
| Expected quantum benefit | 3 | Extends rotamer optimization to more qubits |
| Earliest plausible utility | Medium (5–7 years) | |

**QADF Classification: B**

**Notes**: Biologically important and structurally similar to the rotamer optimization problem (sub-problem 4), but requires more qubits due to the larger interface size. A natural next step after demonstrating quantum advantage on single-chain rotamer optimization — pending hardware scaling.

---

### Subproblem 8: Disordered Region Ensemble Sampling

| Criterion | Score/Value | Notes |
|---|---|---|
| Biological importance | 4 | IDRs involved in signaling, transcription; hard to study |
| Search space discreteness | 1 | Conformational ensemble is continuous and high-dimensional |
| QUBO/Ising compatibility | 1 | Inherently continuous; no natural QUBO mapping |
| Estimated qubit count | >100 | Ensemble sampling requires exponential qubits |
| Gate depth bottleneck | High | |
| Noise sensitivity | High | |
| Classical baseline strength | 3 | MD sampling partially effective but slow |
| Expected quantum benefit | 2 | Quantum walk sampling speculative |
| Earliest plausible utility | Far (>10 years) | |

**QADF Classification: C — Poor near-term target**

**Notes**: AlphaFold assigns low pLDDT to IDRs [REF-14, DOI: 10.1038/s41467-025-56572-9], which are genuinely disordered — the problem is not just prediction difficulty but intrinsic structural multiplicity. Quantum sampling of IDR ensembles would require quantum phase estimation or quantum Monte Carlo algorithms not yet feasible on NISQ hardware.

---

## 2.2 Summary Classification Table

| # | Subproblem | Class | Key Reason |
|---|---|---|---|
| 1 | Global backbone folding | **C** | Continuous space; AF2 near-solved; QAOA fails at p=40 [REF-06] |
| 2 | Short peptide conformational search | **B** | Lattice model only; realistic force fields not QUBO-compatible |
| 3 | Catalytic loop refinement | **B** | Proof-of-concept on QPU [REF-01]; requires real hardware |
| 4 | **Side-chain rotamer optimization** | **A** ⭐ | Discrete Dunbrack library; perfect QUBO fit; 15 qubits at n=5 |
| 5 | Constrained local energy minimization | **C** | Continuous; classical gradient descent dominates |
| 6 | Disulfide bond network optimization | **A** | Perfect matching QUBO; niche scope |
| 7 | Protein-protein interface packing | **B** | Extends rotamer problem; needs more qubits |
| 8 | Disordered region ensemble sampling | **C** | Inherently continuous; no QUBO mapping |

---

## 2.3 QADF Decision Tree (ASCII)

```
START: Is the subproblem's state space DISCRETE and FINITE?
├── NO → [C] Poor near-term target
│   Examples: Global backbone folding, Local energy minimization,
│             IDR ensemble sampling
│   Reason: No QUBO mapping; continuous optimization dominated by
│           classical gradient methods
│
└── YES → Can it be encoded in ≤25 qubits for biologically relevant n?
    ├── NO → [B] Medium-term target
    │   Examples: Short peptide search (realistic), PPI packing
    │   Reason: Feasible QUBO mapping but requires >NISQ scale
    │   Timeline: 5–7 years (pending hardware scaling)
    │
    └── YES → Is the classical baseline weak or computationally hard?
        ├── NO (classical solves it easily) → [B] Medium-term
        │   Example: Disulfide bond optimization (small n, exact matching)
        │
        └── YES → Does pairwise energy structure map to quadratic Ising terms?
            ├── NO → [B] Medium-term (encoding overhead too high)
            │
            └── YES → [A] Near-term hybrid candidate
                Example: ★ SIDE-CHAIN ROTAMER OPTIMIZATION ★
                         (Catalytic loop refinement: borderline A/B)
                Reason: Dunbrack library [REF-05] discretizes chi angles;
                        one-hot encoding → QUBO [REF-02];
                        15 qubits at n=5 residues (NISQ-feasible);
                        COBYLA+CVaR optimizer demonstrated [REF-02];
                        calibrated confidence estimation (this work)

ADDITIONAL FILTER: Is classical baseline already VERY strong?
  Global backbone folding: AF2 GDT_TS = 92.4 → quantum adds no value
  Side-chain rotamers: FlowPacker [REF-07] strong but χ1/χ2 not
                       perfectly recovered; confidence not calibrated
                       → quantum hybrid targets remaining gap

TIMELINE CLASSIFICATION:
  A (Near-term): NISQ hardware (50–100 qubits), classical simulation now
  B (Medium-term): Fault-tolerant ~100–1000 logical qubits (5–10 years)
  C (Long-term / Poor): Full fault-tolerant (>1000 logical qubits, 10+ years)
                        OR fundamentally wrong problem structure for QC
```

---

## 2.4 Rationale for Primary Target Selection

Side-chain rotamer optimization satisfies **all seven QADF selection criteria simultaneously**:

1. ✓ Finite discrete state space (Dunbrack rotamer bins)
2. ✓ Quadratic QUBO formulation (pairwise interaction energies)
3. ✓ Qubit overhead within NISQ simulation range (15 qubits for n=5)
4. ✓ Short circuit depth (QAOA p=1: ~30 CNOT gates for 12-qubit instance)
5. ✓ Biologically critical outcome (binding, stability, function)
6. ✓ Gap in classical methods (χ1/χ2 errors remain at 10–20° for difficult residues)
7. ✓ Calibrated confidence estimation not yet demonstrated in quantum setting

This makes it the uniquely appropriate entry point for near-term quantum-classical hybrid protein structure prediction.

---

*References: REF-01 (DOI: 10.1021/acs.jctc.4c00067), REF-02 (arXiv: 2507.19383), REF-03 (DOI: 10.1371/journal.pcbi.1011033), REF-04 (DOI: 10.1002/prot.26257), REF-05 (DOI: 10.1002/prot.22921), REF-06 (DOI: 10.1038/s41534-023-00733-5), REF-07 (DOI: 10.1101/2024.07.05.602280), REF-11 (NISQ noise characterization), REF-14 (DOI: 10.1038/s41467-025-56572-9)*
