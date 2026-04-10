# PHASE 0 — VERIFIED LITERATURE LOG
# Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)
# Compiled: 2026-04-07
# All entries verified via live tool-based queries.

---

## ANCHOR PAPERS — VERIFIED

### [REF-01] Doga et al. 2024 — JCTC Perspective (CONFIRMED)
- **Title**: A Perspective on Protein Structure Prediction Using Quantum Computers
- **Authors**: Hakan Doga, Bryan Raubenolt, Fabio Cumbo, Jayadev Joshi, Frank P. DiFilippo, Jun Qin, Daniel Blankenberg, Omar Shehab
- **Journal**: J. Chem. Theory Comput. 2024, 20, 9, 3359–3378
- **DOI**: 10.1021/acs.jctc.4c00067
- **Published**: 2024-05-04
- **Key claims**:
  - Framework for systematically selecting protein structure prediction subproblems amenable to quantum advantage
  - 1,321 protein structures in PDB ≤22 residues; ~3,000 sequences ≤41 amino acids (509 with ≥1 mutation)
  - Proof-of-concept: accurately predicted catalytic loop ("P-loop") of Zika Virus NS3 Helicase on IBM 127-qubit Eagle (R3) quantum hardware
  - Ansatz: RealAmplitudes (Qiskit), 1 repetition; metrics: qubit count, circuit depth, ECR depth, measurements
  - Limitations: resource estimation for comprehensive workflow not complete; VQE may not find absolute ground state; qubit connectivity varies by hardware
- **Relevance**: Primary inspiration for QADF framework; directly cited as precedent for subproblem selection

### [REF-02] Agathangelou et al. 2025 — arXiv Side-Chain QAOA (CONFIRMED)
- **Title**: Quantum Algorithm for Protein Side-Chain Optimisation: Comparing Quantum to Classical Methods
- **Authors**: Anastasia Agathangelou (IBM Research Europe – Zurich), Dilhan Manawadu, Ivano Tavernelli (Hartree Centre, STFC)
- **arXiv**: 2507.19383
- **DOI**: https://doi.org/10.48550/arXiv.2507.19383
- **Submitted**: 2025-07-25 (15 pages, 8 figures, 7 tables)
- **Key claims**:
  - QUBO formulation of rotamer optimization → Ising Hamiltonian → QAOA with local XY mixer (Qiskit)
  - Two-body pairwise interaction energies via PyRosetta scoring function
  - Classical benchmark: dual annealing (L-BFGS-B local phase)
  - Optimizer: COBYLA with CVaR (α=0.2)
  - Reduction in computational cost vs. classical simulated annealing
  - Provides crossing-point estimate where quantum method may outperform classical
- **Relevance**: Direct precedent for QUBO encoding of rotamer problem; QAOA with XY mixer; scaling analysis

### [REF-03] Khatami et al. 2023 — Gate-Based Quantum Protein Design (CONFIRMED)
- **Title**: Gate-based quantum computing for protein design
- **Authors**: Mohammad Hassan Khatami, Udson C. Mendes, Nathan Wiebe, Philip M. Kim
- **Journal**: PLOS Computational Biology 2023-04-12
- **DOI**: 10.1371/journal.pcbi.1011033
- **arXiv**: 2201.12459
- **Key claims**:
  - Grover's algorithm circuits for protein design; up to 234 qubits simulated
  - Quadratic speedup over classical search (O(√N) vs O(N))
  - Two models: SP model (lattice-like, integers) and MR model (Coulomb potential, distances)
  - Classical advantage threshold: N>56 designable sites
- **Relevance**: Resource estimation for gate-based protein design; confirms qubit overhead

### [REF-04] Jumper et al. 2021 (CASP14 operations paper) — AlphaFold2 CASP14 (CONFIRMED)
- **Title**: Applying and improving AlphaFold at CASP14
- **Authors**: John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, et al.
- **Journal**: Proteins 2021;89(12):1711–1721
- **DOI**: 10.1002/prot.26257
- **PMC**: PMC9299164
- **KEY PUBLISHED BENCHMARK NUMBERS** (verified, to be used in Phase 8):
  - Median domain GDT_TS: **92.4** (best-of-5, across 92 CASP14 domains)
  - Domains with GDT_TS > 90: **58** out of 92 (best-of-5)
  - Mean domain GDT_TS (all five models): **87.32**; top-1 selection: **88.01**
  - Model ranking by pLDDT achieved 86% of maximum possible improvement
  - T1044 full-sequence TM-score: final system **0.960** (vs. original system **0.807**)
  - T1047s1-D1 failure: best-of-5 GDT_TS = **50.47** (long beta sheet at wrong angle)
  - Side-chain χ1/χ1+2 recovery rates: **NOT REPORTED** in this paper
  - pLDDT calibration: regresses true per-residue lDDT-Cα; low pLDDT indicates real disorder (not always model failure)
  - Source: Table 1A/B (GDT_TS comparisons); Figure 5 (T1064 pLDDT vs. lDDT-Cα correlation)

### [REF-04b] AlphaFold2 Nature Paper (companion)
- **Title**: Highly accurate protein structure prediction with AlphaFold
- **Authors**: Jumper et al.
- **Journal**: Nature 596, 583–589 (2021)
- **DOI**: 10.1038/s41586-021-03819-2
- **Note**: Side-chain χ1/χ2 recovery not prominently reported in Nature paper; AlphaFold 2 focuses on backbone accuracy. Side-chain accuracy reported in literature via CASP14 assessors' analysis and follow-up benchmarks.

### [REF-05] Dunbrack 2011 — Backbone-Dependent Rotamer Library (CONFIRMED)
- **Title**: A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions
- **DOI**: 10.1002/prot.22921 (PubMed: 21645855)
- **Published**: 2011-06-08
- **Key claims**:
  - Rotamer frequencies, mean dihedral angles, and variances as smooth/continuous function of φ,ψ
  - Backbone-dependent; used in Rosetta as scoring function: E = -ln(p(rotamer_i | φ,ψ))
  - Adaptive kernel density estimation
- **Homepage**: https://dunbrack.fccc.edu/lab/bbdep2010
- **Relevance**: Ground-truth rotamer definitions and prior probabilities for QUBO encoding

---

## SUPPORTING PAPERS — VERIFIED

### [REF-06] Bauza et al. 2023 — QAOA Peptide Conformational Sampling (CONFIRMED)
- **Title**: Peptide conformational sampling using the Quantum Approximate Optimization Algorithm
- **Authors**: (multiple; Nature npj Quantum Information)
- **Journal**: npj Quantum Information 9, 68 (2023)
- **DOI**: 10.1038/s41534-023-00733-5
- **arXiv**: 2204.01821
- **Key claims (honest/negative results)**:
  - QAOA benchmarked on alanine tetrapeptide folding (Lennard-Jones, lattice model)
  - Up to 28 qubits used in numerical experiments [CLASSICALLY SIMULATED]
  - For self-avoiding walks: QAOA provides exponentially growing advantage over uniform random sampling at fixed depth
  - For full peptide folding (realistic potential): >40 ansatz layers needed to approach minimum energy; QAOA performance matchable by random sampling with small overhead (max ratio 6.6×)
  - Conclusion: "These results cast serious doubt on the ability of QAOA to address the protein folding problem in the near term, even in an extremely simplified setting."
  - Mixed findings suggest QAOA better for constraint satisfaction than continuous/mixed optimization
- **Relevance**: Critical honest assessment of QAOA limitations; supports QADF classification of global backbone folding as poor near-term target

### [REF-07] FlowPacker 2024 — Classical Side-Chain Baseline (CONFIRMED)
- **Title**: FlowPacker: Protein side-chain packing with torsional flow matching
- **Authors**: (MJ Lee et al.)
- **Source**: bioRxiv 2024.07.05.602280; NeurIPS MLSB 2024 workshop
- **DOI**: 10.1101/2024.07.05.602280
- **Key claims**:
  - Torsional flow matching + equivariant graph attention for side-chain prediction
  - Outperforms previous state-of-the-art baselines across most metrics with improved runtime
  - Works for inpainting, multimeric targets, antibody-antigen complexes
  - Code: https://gitlab.com/mjslee0921/flowpacker
- **Relevance**: State-of-the-art classical baseline for side-chain packing (used in comparison)

### [REF-08] CalPro 2026 — Calibrated Uncertainty for Protein Structure (CONFIRMED)
- **Title**: CalPro: Prior-Aware Evidential–Conformal Prediction with Structure-Aware Coverage Guarantees
- **arXiv**: 2601.07201 (2026-01-12)
- **Key claims**:
  - pLDDT is NOT a calibrated probability; systematic miscalibration under distribution shift (15–25% coverage degradation for baselines)
  - CalPro achieves ≤5% coverage degradation vs. 15–25% for baselines
  - Reduces calibration error by 30–50%; improves ligand-docking success by 25%
  - Pairwise evidential-conformal framework; Normal-Inverse-Gamma head
- **Relevance**: Confirms pLDDT miscalibration; motivates calibrated confidence in this project

### [REF-09] Uncertainty Quantification in Drug Discovery 2025 (CONFIRMED)
- **Title**: Uncertainty quantification enables reliable deep learning for protein-ligand binding affinity
- **Journal**: Nature Scientific Reports 2025
- **DOI**: 10.1038/s41598-025-27167-7
- **Key methods**: MC Dropout, Deep Ensemble, Bayes by Backprop, Laplace, Evidential NN; ECE/RMSCE/MACE calibration metrics

### [REF-10] RCSB PDB Data API (CONFIRMED)
- **Source**: https://data.rcsb.org ; https://www.rcsb.org/docs/programmatic-access/file-download-services
- **Key facts**: REST API at https://data.rcsb.org/rest/v1/core/entry/{PDB_ID}; Python rcsb-api package; HTTPS downloads at files.wwpdb.org
- **Relevance**: Programmatic PDB access for data implementation

### [REF-11] NISQ Noise Characterization — Emergent Mind Review (CONFIRMED)
- **Source**: emergentmind.com/topics/noisy-intermediate-scale-quantum-nisq-hardware (Dec 2025)
- **Key values (verified)**:
  - Single-qubit gate error: ε₁ ~ 10⁻⁴ – 10⁻³
  - Two-qubit (CNOT) gate error: ε₂ ~ 10⁻³ – 10⁻²
  - T1, T2 coherence: 20–200 μs (superconducting transmons)
  - Readout infidelity: 1–5% per measurement
  - Depolarizing channel standard model for noise simulation
- **Relevance**: Justifies noise parameter choices in Phase 6 noise analysis

### [REF-12] Hybrid VQC Machine Learning 2025 (CONFIRMED)
- **Title**: A Framework for Hybrid Quantum Classical Machine Learning
- **arXiv**: 2502.11951 (2025-02-17)
- **Key claims**: Amplitude/angle encoding for classical-to-quantum data pipeline; VQC as trainable feature transformer

### [REF-13] AlphaFold pLDDT Color Convention (CONFIRMED from multiple sources)
- **Color scale** (official AlphaFold pLDDT convention, confirmed via AF database documentation):
  - >90: Dark blue (#0053D6) — Very high confidence
  - 70–90: Light blue (#65CBF3) — Confident
  - 50–70: Yellow (#FFDB13) — Low confidence
  - <50: Orange (#FF7D45) — Very low confidence
- **Sources**: AlphaFold protein structure database; ailienamaggiolo/alphafold_coloring GitHub; Leipzig University PyMOL tutorial; YouTube AlphaFold interpretation guide
- **Note**: pLDDT stored in B-factor column of PDB files; regresses true lDDT-Cα

### [REF-14] AlphaFold IDR / Disordered Regions (CONFIRMED)
- **Source**: Nature Communications 2025 (AlphaFold-Metainference, doi: 10.1038/s41467-025-56572-9)
- **Key claim**: Long regions of low pLDDT indicate disordered regions not well predicted by single-structure AlphaFold; AF-Metainference uses AF distances as restraints in MD simulations
- **Relevance**: Confirms AlphaFold weakness in IDRs; motivates hybrid approach targeting low-pLDDT subproblems

### [REF-15] EGNN Protein Representation (CONFIRMED)
- **Source**: ENGINE paper — Genome Biology 2025 (PMC12665208)
- **Key**: EGNN maintains SE(3) equivariance; EGCL updates node coordinates and features; K-nearest neighbor graph construction
- **Relevance**: Justification for equivariant GNN as classical backbone in Phase 3

---

## UNVERIFIED / REJECTED ENTRIES

| Paper | Status | Reason | Substitute |
|-------|--------|--------|------------|
| AlphaFold2 χ1/χ2 side-chain recovery in Nature paper | NOT FOUND in primary paper | AF2 Nature paper focuses on backbone metrics | Use follow-up CASP14 assessors' analysis or FlowPacker benchmarks |
| QPacker (D-Wave side-chain, Rosetta integration) | Cited in [REF-02] as ref [27] but not independently verified via DOI | Cannot confirm DOI without paywall | Mentioned as prior work via [REF-02] reference list only |

---

## PHASE 0 CHECKPOINT

**Completed**: Literature verification via live web queries. All anchor papers confirmed with DOI or arXiv ID. Key benchmark numbers retrieved and sourced.

**Files written**: /results/logs/phase0_literature_verified.md

**Key decisions affecting downstream work**:
1. Primary task confirmed as **side-chain rotamer optimization** — strong literature support (REF-01, REF-02)
2. AlphaFold χ1/χ2 side-chain recovery NOT in primary Nature/CASP14 paper — comparison table must note this limitation explicitly
3. QAOA on full protein folding: negative results documented (REF-06) — confirms QADF should classify global folding as poor near-term target
4. FlowPacker (REF-07) is the state-of-the-art classical side-chain baseline to benchmark against
5. pLDDT is NOT calibrated (REF-08) — justifies calibrated uncertainty as a contribution

**Next phase requires**: Verified reference list (this file); all Phase 1 claims must cite REF-01 through REF-15 only.
