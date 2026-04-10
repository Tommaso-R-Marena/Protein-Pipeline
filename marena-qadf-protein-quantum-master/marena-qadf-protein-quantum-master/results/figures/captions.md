# Figure Captions
## QADF Project: Hybrid Quantum-Classical Protein Structure Prediction

**Fig 1**: Hybrid quantum-classical protein side-chain rotamer optimization pipeline. Input PDB structure features are processed by an equivariant GNN (EGNN) classical backbone, passed through a parameterized quantum circuit (PQC) feature transformer [CLASSICALLY SIMULATED], encoded as a QUBO instance, and optimized with QAOA [CLASSICALLY SIMULATED]. Output: per-residue rotamer class probabilities and calibrated confidence scores.

---

**Fig 2**: QADF subproblem scoring table for 8 protein structure prediction tasks. Columns: discreteness, QUBO compatibility, qubit count (n=5 residues), gate depth, noise sensitivity, classical baseline strength, expected quantum benefit, timeline, and classification (A/B/C). Side-chain rotamer optimization (★) is the only subproblem classified A with consistent high scores across all criteria relevant to near-term quantum hardware.

---

**Fig 3**: Left — QAOA simulation time vs. window size (n residues) [CLASSICALLY SIMULATED]. Exponential scaling is apparent; the red dashed line marks the practical feasibility boundary (~6 residues, 18 qubits) for classical statevector simulation within the time budget. Right — Qubit count required (3n for n residues with 3 rotamer states). Blue bars: tractable on NISQ hardware; red bars: exceed ~20-qubit limit for near-term devices.

---

**Fig 4**: QAOA noise analysis [CLASSICALLY SIMULATED] under depolarizing channel with error rates ε₂ ∈ {0, 10⁻³, 10⁻²} (left: objective value; right: percent degradation). At the lower end of the NISQ error range (ε₂ = 10⁻³), energy degrades by ~2.6%. At the upper end (ε₂ = 10⁻²), degradation reaches ~23.3%, highlighting the sensitivity of QAOA to gate errors. Noise parameters chosen to match published NISQ two-qubit gate error rates (REF-11).

---

**Fig 5**: Reliability diagram (calibration plot) for per-residue confidence predictions on 1L2Y. The diagonal dashed line represents perfect calibration. All 17 chi1-bearing residues fall in the high-confidence bin (>80%), with 100% rotamer bin accuracy. ECE = 0.0148 ± 0.0045 (95% CI: [0.0072, 0.0242]), indicating good calibration on this small dataset. A larger dataset would enable finer binning and more discriminative calibration analysis.

---

**Fig 6**: Scatter plot of per-residue confidence score vs. absolute chi1 dihedral error (degrees) for 1L2Y. Points are colored by the AlphaFold pLDDT color convention (REF-13): dark blue (>90), light blue (70-90), yellow (50-70), orange (<50). Pearson r = −0.310 (expected sign: negative). Residues with large chi1 errors (TYR3, GLN5, ASP9, PRO12) are labeled. The trend is consistent with confidence scores providing meaningful uncertainty information, though the small dataset limits statistical power.

---

**Fig 7**: Per-residue confidence profile for 1L2Y (20-residue TC5b Trp-cage). Bars colored by AlphaFold pLDDT convention (REF-13): dark blue (>90), light blue (70-90), yellow (50-70), orange (<50). Glycine residues (G10, G11, G15) receive lower confidence scores because they lack chi1 angles, making the discrete rotamer prediction task ill-defined. Most side-chain-bearing residues are predicted with high confidence (>90), consistent with the small, well-ordered NMR structure of 1L2Y.

---

**Fig 8**: QAOA convergence histories [CLASSICALLY SIMULATED] for p=1 (left) and p=2 (right), optimized with COBYLA (200 iterations, 12 qubits, 4-residue 1L2Y window). The Ising Hamiltonian expectation value decreases monotonically, indicating successful optimization of the variational parameters. p=1 achieves faster convergence; p=2 has a larger parameter space (4 vs. 2 parameters) but a deeper circuit (depth ~312 vs. ~156 CNOT gates).

---

**Fig 9**: Left — solution quality comparison (QUBO objective value, lower=better) for exhaustive search, greedy assignment, simulated annealing, QAOA p=1 [CS], and QAOA p=2 [CS] on the 4-residue 1L2Y window. Classical methods (exhaustive, greedy, SA) all find the optimal solution (−34.07). QAOA at p=1,2 does not reach the optimal at these low circuit depths, consistent with known QAOA limitations (REF-06). Right — runtime comparison. Classical methods are faster for this small instance; QAOA runtimes reflect classical simulation overhead, not hardware runtime. [CS] = Classically Simulated.

---

**Fig 10**: Left — this project's confidence score distribution for 1L2Y (20 residues). Most residues receive high confidence (>90, dark blue), with Glycine residues (no chi1) receiving lower scores. Right — SCHEMATIC qualitative comparison with a typical AlphaFold 2 pLDDT distribution (not from primary data; for visual shape comparison only). AF2 pLDDT distributions typically show a bimodal pattern: a high-confidence peak for structured regions and a lower-pLDDT tail for disordered regions (IDRs). This project's confidence scores are calibrated to chi1 rotamer accuracy rather than backbone lDDT-Cα, making direct numerical comparison inappropriate.

---

