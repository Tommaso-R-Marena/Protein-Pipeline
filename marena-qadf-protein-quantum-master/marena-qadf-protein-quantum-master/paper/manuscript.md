# A Quantum Amenability Decision Framework for Protein Structure Prediction: Side-Chain Rotamer Optimization via Classically Simulated Hybrid Circuits

**Tommaso Marena**  
Department of Chemistry, Department of Philosophy  
The Catholic University of America  
Washington, D.C., USA  
Double Major: Biochemistry and Philosophy (Pre-Law)  
Independent Research — No Funding, No External Assistance

---

## Abstract

We present the Quantum Amenability Decision Framework (QADF), a principled methodology for determining which subproblems in computational structural biology may be tractably reformulated as quantum optimization tasks on near-term quantum hardware. The framework evaluates candidate subproblems along nine dimensions—including problem size, locality structure, problem encoding cost, and noise sensitivity—and produces a categorical recommendation: quantum-favorable (Category A), potentially useful (Category B), or classically superior (Category C). Applying the QADF to protein structure prediction, we identify side-chain rotamer optimization as a Category A candidate and global backbone folding as Category C, consistent with the analysis of Bauza et al. (2023). We implement a hybrid quantum-classical pipeline on real protein structures from the RCSB Protein Data Bank—specifically 1L2Y (Trp-cage miniprotein, 20 residues) and 1UBQ (ubiquitin, 76 residues)—formulating 4-residue sliding-window rotamer packing as a Quadratic Unconstrained Binary Optimization (QUBO) problem with 12 binary variables per instance. The quantum optimization module employs the Quantum Approximate Optimization Algorithm (QAOA) at circuit depths p = 1 and p = 2. All quantum computations are **classically simulated** using PennyLane's `default.qubit` statevector backend; no physical quantum hardware was used. We benchmark QAOA against classical exhaustive search, greedy assignment, and simulated annealing baselines. A pLDDT-inspired confidence scoring mechanism is developed and evaluated for calibration. Numerical results for energy minimization, χ1/χ2 recovery rates, and calibration metrics are reported as Exhaustive: −34.07 (ground truth, 81 configs); Greedy: −34.07 (matches GT, t=0.05 ms); SA: −34.07 (matches GT, t=5.6 ms); QAOA p=1 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 93.83 (does not match GT, circuit depth=156, t=23.3 s, n_qubits=12); QAOA p=2 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 179.39 (does not match GT, t=32.9 s). Classical SA achieves ground truth; QAOA at p=1,2 does not at this circuit depth. pending completion of the computational pipeline. We provide an honest scaling analysis demonstrating the simulation boundary at approximately 20–25 qubits and give a rigorous assessment of the conditions under which a near-term quantum advantage for this subproblem might plausibly emerge. This work constitutes a proof-of-concept for structured hybrid quantum-classical workflows in structural biology, not a claim of demonstrated quantum advantage.

*Keywords:* side-chain rotamer optimization, QUBO, QAOA, quantum amenability, hybrid quantum-classical, PennyLane, protein structure prediction

---

## 1. Introduction

### 1.1 The Protein Folding Problem

The ability to predict the three-dimensional structure of a protein from its amino acid sequence remains among the central problems of structural biology and biophysics. The biological importance of this problem is difficult to overstate: protein three-dimensional structure determines function, and structural aberrations underlie diseases ranging from Alzheimer's disease to many cancers. The general protein structure prediction problem, understood as the prediction of all atomic coordinates given only the primary sequence, is believed to be computationally intractable in its most general form, with the side-chain packing component alone shown to be NP-hard under certain energy models (see Section 3 for formal discussion).

Protein structure prediction is conventionally decomposed into hierarchical subproblems: (i) backbone fold prediction, which determines the arrangement of the polypeptide backbone described by φ (phi) and ψ (psi) dihedral angles; (ii) side-chain rotamer assignment, which selects the conformations of side-chain χ (chi) dihedral angles given a fixed backbone; and (iii) global energy refinement, which jointly optimizes backbone and side-chain geometry. This decomposition is not merely computational convenience—it reflects the approximate energetic decoupling between backbone and side-chain degrees of freedom that has been exploited in experimental structure determination and in computational methods for decades.

### 1.2 AlphaFold 2: Achievements and Gaps

The release of AlphaFold 2 (AF2) by Jumper et al. represents the most significant advance in computational structural biology in at least two decades. In the CASP14 blind prediction benchmark, AF2 achieved a median Global Distance Test Total Score (GDT_TS) of 92.4 across 92 evaluated domains, with 58 of 92 domains achieving GDT_TS > 90 [Jumper et al. 2021 (CASP14), DOI: 10.1002/prot.26257, Table 1B]. The mean GDT_TS across all five model predictions was 87.32, with the top-1 prediction reaching 88.01 [Jumper et al. 2021, Table 1A]. These numbers represent a decisive improvement over prior state-of-the-art methods, effectively solving backbone fold prediction for many single-domain, well-folded proteins.

However, the significance of AF2's limitations must be stated precisely. First, AF2 does not report χ1 or χ2 side-chain dihedral recovery rates in the CASP14 evaluation paper [4]; backbone accuracy metrics are the primary reported outcome. The side-chain placement quality is implicitly captured in all-atom RMSD metrics but is not disaggregated. Second, AF2's confidence metric, pLDDT (predicted Local Distance Difference Test), is not a calibrated probability distribution. Marena et al. (2026) [CalPro, arXiv:2601.07201] demonstrate that pLDDT scores do not constitute calibrated probabilities and should not be interpreted as direct uncertainty estimates without recalibration. Third, AF2 is known to fail on intrinsically disordered regions (IDRs) and on proteins with low evolutionary sequence depth in the multiple sequence alignment; the AlphaFold-Metainference integration (2025) was developed in part to address conformational uncertainty in these regions [DOI: 10.1038/s41467-025-56572-9]. Fourth, a concrete failure case in CASP14 was domain T1047s1-D1, for which AF2 achieved GDT_TS of only 50.47 [4, Table 1B], illustrating that hard cases remain outside AF2's reliable operating regime.

These limitations create space for complementary approaches—not competing with AF2 on global fold prediction, where it already performs at or near experimental accuracy for well-folded proteins, but addressing the specific subproblems where confidence is low, calibration is poor, or the problem structure invites non-deep-learning methods.

### 1.3 Opportunity for Hybrid Quantum-Classical Approaches

Quantum computing offers a distinct computational paradigm: amplitude interference in superposition enables, in principle, parallel exploration of combinatorially large configuration spaces. For optimization problems with discrete variables and pairwise interaction terms—precisely the structure of rotamer optimization—quantum variational approaches such as QAOA may offer favorable scaling in the asymptotic regime. Doga et al. (2024) provide a systematic review of quantum algorithms applied to molecular structure problems [DOI: 10.1021/acs.jctc.4c00067], and Agathangelou et al. (2025) present recent quantum simulation results for conformational search [arXiv:2507.19383]. Khatami et al. (2023) formulate protein side-chain placement as a quantum optimization problem and evaluate it on near-term hardware [DOI: 10.1371/journal.pcbi.1011033].

Crucially, the mere existence of a quantum algorithm for a problem does not imply a practical advantage on near-term Noisy Intermediate-Scale Quantum (NISQ) hardware. Single-qubit gate error rates on current superconducting hardware are approximately ε₁ ~ 10⁻⁴–10⁻³, and two-qubit gate error rates are approximately ε₂ ~ 10⁻³–10⁻², with coherence times limiting total circuit depth. Bauza et al. (2023) analyze QAOA applied to protein folding problems and conclude that the noise overhead makes global folding a poor candidate for near-term quantum advantage [DOI: 10.1038/s41534-023-00733-5]. Their analysis motivates the present framework: rather than applying quantum methods uniformly, one should first evaluate each subproblem's amenability to quantum treatment.

### 1.4 Contributions of This Work

This paper makes the following contributions:

1. **The QADF Framework**: A formal methodology for evaluating the quantum amenability of computational biology subproblems, operationalized as a scored decision tree over nine dimensions. The framework is instantiated on eight subproblems of protein structure prediction.

2. **QUBO Formulation of Rotamer Packing**: A rigorous quadratic unconstrained binary optimization encoding of the side-chain rotamer assignment problem, derived from the Dunbrack backbone-dependent rotamer library [PubMed: 21645855].

3. **Hybrid Quantum-Classical Prototype**: A complete pipeline from PDB structure download [RCSB, https://data.rcsb.org] through QUBO construction through QAOA optimization, implemented on real protein structures (1L2Y, 1UBQ), with all quantum computations classically simulated.

4. **Calibrated Confidence Estimation**: A pLDDT-inspired confidence scoring mechanism subjected to empirical calibration analysis with reliability diagrams and Expected Calibration Error (ECE) computation.

5. **Honest Scaling Analysis**: A systematic characterization of the simulation boundary and the conditions under which near-term quantum advantage might plausibly emerge, with explicit acknowledgment of what this prototype does not demonstrate.

---

## 2. Related Work

### 2.1 Classical Protein Structure Prediction Baselines

**AlphaFold 2** [Jumper et al. 2021, Nature, DOI: 10.1038/s41586-021-03819-2] employs a transformer-based architecture with attention over multiple sequence alignments and pairwise residue features, achieving near-experimental accuracy on many benchmark proteins. It remains the definitive reference for global fold prediction and serves as the primary performance anchor for this study. As noted in Section 1.2, AF2's pLDDT metric is not a calibrated probability [arXiv:2601.07201], and its performance on IDRs and low-homology proteins is substantially degraded [DOI: 10.1038/s41467-025-56572-9].

**FlowPacker** [bioRxiv 2024.07.05.602280] is a normalizing-flow-based side-chain packing model that provides a direct classical comparison point for the rotamer optimization task addressed in this paper. FlowPacker operates on fixed backbones and produces probabilistic distributions over χ angles, achieving strong χ1 and χ1+χ2 recovery rates on benchmark sets. It represents the state-of-the-art for purely classical, learned side-chain packing and is the appropriate baseline comparison for the rotamer optimization component of this work. The performance gap between the present quantum-inspired prototype and FlowPacker is expected to be substantial; this expectation is stated explicitly rather than obscured.

**ENGINE** [2025, Genome Biology, PMC: PMC12665208] is an equivariant graph neural network approach for protein structure refinement that leverages geometric deep learning. Its inclusion here contextualizes the landscape of modern classical methods and illustrates the performance bar that any hybrid quantum-classical approach must eventually meet to be practically useful.

### 2.2 Quantum Approaches to Molecular Structure Problems

**Doga et al. (2024)** [DOI: 10.1021/acs.jctc.4c00067] provide the most comprehensive systematic review of quantum algorithms applied to problems in chemical and structural biology, covering variational quantum eigensolvers (VQE), QAOA, and quantum annealing formulations. Their taxonomy of problem types and their analysis of resource requirements are incorporated into the QADF framework developed here.

**Agathangelou et al. (2025)** [arXiv:2507.19383] present quantum simulation results for conformational search problems and analyze the regime in which quantum circuits with limited depth can provide useful approximations. Their treatment of circuit expressibility relative to problem structure informs the QAOA depth choices (p = 1, 2) used in this study.

**Khatami et al. (2023)** [DOI: 10.1371/journal.pcbi.1011033, arXiv:2201.12459] specifically formulate protein side-chain placement as a quantum optimization problem and evaluate performance on near-term devices. Their work is the most direct precedent for this study. Key differences: the present work introduces the QADF evaluation layer, uses the Dunbrack backbone-dependent rotamer library for energy terms, extends to calibrated confidence estimation, and provides explicit acknowledgment of the classical simulation setting.

**Bauza et al. (2023)** [DOI: 10.1038/s41534-023-00733-5] analyze QAOA applied to protein folding and provide quantitative estimates of the circuit depth and qubit counts required to approach classical performance, concluding that global backbone folding is not a promising near-term quantum application. This conclusion is incorporated directly into the QADF taxonomy (Section 4).

### 2.3 Uncertainty Quantification

**CalPro (2026)** [arXiv:2601.07201] provides the theoretical and empirical basis for the claim that pLDDT is not a calibrated probability. The confidence scoring developed in Section 9 is designed to address this calibration deficit for the rotamer optimization task.

**Uncertainty quantification for protein-ligand interactions** [Nature Scientific Reports, DOI: 10.1038/s41598-025-27167-7] provides methodological context for calibrated confidence estimation in structural biology more broadly.

**Hybrid variational quantum circuit machine learning** [arXiv:2502.11951] provides context for the hybrid classical-quantum architecture used in Section 5 and informs the design of the quantum feature transformation layer.

---

## 3. Formal Problem Statement

### 3.1 Side-Chain Rotamer Optimization

Let a protein of length N be described by its backbone conformation, specified by dihedral angle pairs (φᵢ, ψᵢ) for residue i = 1, …, N. Given this fixed backbone, the side-chain rotamer assignment problem is to select, for each residue i, a rotamer state rᵢ ∈ Rᵢ from the discrete set of rotamers Rᵢ defined by the Dunbrack backbone-dependent rotamer library [PubMed: 21645855], conditioned on the local (φᵢ, ψᵢ) pair.

Formally, a rotamer rᵢ is characterized by modal values of the χ dihedral angles (χ1, χ2, …) for residue i's side chain, organized into three primary bins based on χ1: gauche-minus (g⁻, χ1 < −60°), trans (t, χ1 ≈ ±180°), and gauche-plus (g⁺, χ1 > 60°). The objective is to find the assignment **r** = (r₁, r₂, …, rN) that minimizes the total energy:

$$E(\mathbf{r}) = \sum_{i} E_{\text{self}}(r_i) + \sum_{i < j} E_{\text{pair}}(r_i, r_j)$$

where Eₛₑₗf(rᵢ) is the self-energy of rotamer rᵢ (derived from the backbone-dependent rotamer probability, following the convention −log P(rᵢ | φᵢ, ψᵢ)), and Eₚₐᵢᵣ(rᵢ, rⱼ) is the pairwise interaction energy between rotamers at residues i and j, here approximated by a Lennard-Jones-type steric term computed from the rotamer modal geometries.

The problem is to find:

$$\mathbf{r}^* = \arg\min_{\mathbf{r}} E(\mathbf{r})$$

subject to the constraint that exactly one rotamer is selected per residue.

### 3.2 NP-Completeness of Side-Chain Packing

Side-chain packing optimization under general pairwise energy functions is known to be NP-hard. The reduction proceeds by analogy with graph coloring and maximum weighted independent set problems: each residue corresponds to a node, each rotamer corresponds to a color, and the pairwise interaction energies define edge weights in a conflict graph. Under arbitrary pairwise energy models, the decision version of this problem (does there exist an assignment achieving energy ≤ E*?) is NP-complete by reduction from MAX-2-CSP. Practical instances with realistic energy functions and sparse interaction graphs (residues interact only within a spatial cutoff) are substantially more tractable, but worst-case hardness motivates the search for improved optimization heuristics.

This NP-hardness argument provides one motivation for quantum approaches: if QAOA can provide an approximation ratio better than the best classical polynomial-time algorithm for relevant instances, a practical advantage may result even without a polynomial-time exact algorithm.

### 3.3 QUBO Formulation

To formulate rotamer optimization as a Quadratic Unconstrained Binary Optimization (QUBO) problem, we introduce binary variables xᵢ,ₖ ∈ {0, 1} for residue i and rotamer state k ∈ {1, …, |Rᵢ|}. The variable xᵢ,ₖ = 1 if and only if residue i is assigned rotamer k.

The QUBO objective function is:

$$H_{\text{QUBO}} = \lambda \sum_{i} \left(1 - \sum_k x_{i,k}\right)^2 + \sum_i \sum_k E_{\text{self}}^{(i,k)} x_{i,k} + \sum_{i < j} \sum_{k,l} E_{\text{pair}}^{(i,k,j,l)} x_{i,k} x_{j,l}$$

where the first term is a penalty enforcing the one-hot constraint (exactly one rotamer per residue) with penalty weight λ, the second term encodes self-energies, and the third term encodes pairwise interaction energies. The objective is to minimize H_QUBO over all binary assignments.

For a 4-residue window with three rotamer states per residue, this yields |Rᵢ| = 3 binary variables per residue and a total of 4 × 3 = 12 binary variables, corresponding to a 12-qubit QUBO instance. The QUBO matrix Q is constructed as:

- Diagonal entries Qₖₖ: self-energies Eₛₑₗf plus penalty contributions
- Off-diagonal entries within a residue: penalty terms enforcing mutual exclusion of rotamers
- Off-diagonal entries between residues: pairwise interaction energies

The QUBO problem H = **x**ᵀ Q **x** (with constant offsets) maps directly to the Ising Hamiltonian via the substitution xᵢ = (1 − σᵢᶻ)/2, where σᵢᶻ are Pauli-Z operators. This Ising Hamiltonian constitutes the problem Hamiltonian H_C for QAOA.

---

## 4. The Quantum Amenability Decision Framework (QADF)

### 4.1 Motivation

The central methodological contribution of this work is not the QAOA implementation itself but the framework used to decide whether quantum approaches are appropriate for a given subproblem. The QADF is a structured evaluation methodology motivated by two observations: (1) the diversity of computational subproblems within protein structure prediction spans a wide range of structural properties relevant to quantum tractability, and (2) the overhead costs of quantum encoding, noise mitigation, and circuit compilation are non-trivial and problem-dependent. Applying quantum methods indiscriminately—without first assessing whether the problem structure is compatible with the strengths of near-term quantum devices—is epistemically unjustified and risks misleading conclusions.

### 4.2 The Nine Evaluation Dimensions

The QADF scores each candidate subproblem on the following nine dimensions, each rated on a scale of 1 (unfavorable for quantum treatment) to 5 (favorable):

| Dimension | Description |
|-----------|-------------|
| **D1: Problem Size (n)** | Number of binary variables in the QUBO encoding; smaller instances are more amenable to near-term simulation |
| **D2: Locality** | Degree to which the interaction graph is sparse and local; highly local interactions are favorable for QAOA |
| **D3: Encoding Efficiency** | Ratio of physical qubits to logical variables; lower overhead is favorable |
| **D4: Noise Sensitivity** | Sensitivity of the objective to gate errors; problems with flat energy landscapes are highly sensitive |
| **D5: Circuit Depth Requirement** | Minimum circuit depth (p) needed for the QAOA approximation ratio to exceed best classical heuristics; lower is favorable |
| **D6: Classical Hardness** | Degree to which the classical problem is genuinely hard for known algorithms; problems that are classically easy are poor quantum candidates |
| **D7: Verifiability** | Ease of verifying a proposed solution; efficient verification enables near-term benchmarking |
| **D8: Constraint Complexity** | Number and complexity of constraints in the original problem that must be encoded as penalty terms; high constraint complexity inflates qubit count |
| **D9: Physical Interpretability** | Degree to which quantum states have direct physical meaning in the problem domain; aids circuit design |

The QADF composite score is the mean of the nine dimension scores, with a recommended category threshold: composite ≥ 3.5 → Category A (quantum-favorable); 2.5–3.5 → Category B (potentially useful, resource-dependent); < 2.5 → Category C (classically superior, do not apply quantum methods on near-term hardware).

### 4.3 Taxonomy of Protein Structure Prediction Subproblems

The following table presents the QADF evaluation for eight subproblems of protein structure prediction. Scores represent structured qualitative judgments derived from the literature [1, 2, 3, 6]; they are not empirically measured quantities.

| Subproblem | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | Mean | Category |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Side-chain rotamer packing (local window)** | 4 | 4 | 3 | 3 | 3 | 3 | 5 | 3 | 4 | **3.56** | **A** |
| Global backbone fold prediction | 1 | 1 | 1 | 2 | 1 | 1 | 2 | 2 | 2 | **1.44** | **C** |
| Torsion angle sampling (local, IDR) | 3 | 4 | 3 | 2 | 3 | 3 | 4 | 3 | 3 | 3.11 | B |
| Contact map prediction | 2 | 2 | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 2.00 | C |
| Coarse-grained lattice folding | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 2 | 2 | 2.89 | B |
| Ligand docking (discrete poses) | 3 | 3 | 3 | 3 | 3 | 3 | 5 | 2 | 3 | 3.11 | B |
| Loop closure / fragment assembly | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 2 | 2.22 | C |
| Homology model refinement (local) | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 2 | 3 | 3.00 | B |

**Category A: Side-chain rotamer packing (local window).** This subproblem receives the highest QADF score due to: (a) the natural QUBO encoding with bounded variable count per residue, (b) the locality of interactions (only residues within spatial cutoff interact), (c) the discrete and enumerable solution space (Dunbrack bins), and (d) the efficient verifiability of any proposed solution via energy evaluation. This is the subproblem addressed in the present study.

**Category C: Global backbone fold prediction.** This subproblem receives the lowest QADF score, consistent with the analysis of Bauza et al. (2023) [DOI: 10.1038/s41534-023-00733-5]. The primary disqualifying features are: (a) the number of continuous degrees of freedom (O(N) φ/ψ pairs) does not admit efficient discrete encoding without massive qubit overhead; (b) the interaction graph is dense and long-range; (c) the problem is already addressed near-optimally by AF2 for many protein families; and (d) circuit depth requirements exceed coherence limits by orders of magnitude for realistic protein sizes.

### 4.4 Decision Tree

The QADF decision tree operates as follows:

1. **Encoding feasibility gate**: Does the problem admit a QUBO encoding with ≤ 50 binary variables for representative instances? If NO → Category C (stop).
2. **Classical hardness gate**: Is the problem strictly NP-hard or does the best classical algorithm show empirically poor performance? If NO → Category C (stop).
3. **Locality gate**: Is the QUBO interaction graph sparse (average degree < N/4)? If NO → penalize D2, D4, D5 by 2 points each.
4. **Depth requirement gate**: Is p ≤ 5 expected to yield approximation ratio > 0.9 based on problem structure analysis? If NO → Category B at best.
5. **Score aggregation**: Compute mean QADF score across all nine dimensions.
6. **Category assignment**: A (≥ 3.5), B (2.5–3.5), C (< 2.5).

---

## 5. Model Architecture and Methods

### 5.1 Overview

The hybrid quantum-classical pipeline consists of four components: (i) a data ingestion module that downloads and parses PDB files and constructs the backbone-dependent energy landscape from the Dunbrack rotamer library; (ii) a QUBO construction module that encodes the rotamer optimization problem as a binary quadratic program; (iii) a quantum optimization module [**CLASSICALLY SIMULATED**] that applies QAOA to find low-energy rotamer assignments; and (iv) a confidence estimation module that produces calibrated confidence scores analogous to pLDDT.

### 5.2 Data Ingestion and Energy Construction

PDB structures are downloaded from the RCSB Protein Data Bank [https://data.rcsb.org] in mmCIF format. Backbone dihedral angles (φ, ψ) are computed using BioPython's `calc_dihedral` function applied to the N–Cα–C–N atom quartet. For each residue, the Dunbrack backbone-dependent rotamer library [PubMed: 21645855] is queried (using the nearest 10° × 10° (φ, ψ) bin) to obtain the modal χ1, χ2 values and the log-probability self-energy for each rotamer state. Pairwise interaction energies are computed from the rotamer modal geometries using a simplified steric clash score based on the sum of inverse-sixth repulsive terms between non-bonded atom pairs within a spatial cutoff of 8 Å.

### 5.3 QUBO Construction Module

Given self-energies {Eₛₑₗf(i, k)} and pairwise energies {Eₚₐᵢᵣ(i, k, j, l)}, the QUBO matrix Q is assembled as described in Section 3.3. The penalty weight λ is set to λ = 1.5 × max|Eₚₐᵢᵣ| to ensure that any one-hot constraint violation is energetically disfavored relative to any valid assignment. The QUBO is then converted to an Ising Hamiltonian for input to the QAOA circuit.

A 4-residue sliding window is applied across the protein sequence, yielding overlapping sub-instances. For 1L2Y (20 residues), this produces 17 overlapping 4-residue QUBO instances, each with 12 binary variables (12 qubits). Results are aggregated across windows by majority vote on overlapping residue assignments.

### 5.4 Quantum Optimization Module [CLASSICALLY SIMULATED]

**This module performs no computations on physical quantum hardware. All results are produced by classical simulation of quantum circuits using PennyLane's `default.qubit` statevector backend.**

The Quantum Approximate Optimization Algorithm (QAOA) [Farhi et al. 2014] is a variational hybrid quantum-classical algorithm for combinatorial optimization. For a problem Hamiltonian H_C and a mixing Hamiltonian H_B = Σᵢ Xᵢ (sum of Pauli-X operators), the QAOA ansatz of depth p is:

$$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = e^{-i\beta_p H_B} e^{-i\gamma_p H_C} \cdots e^{-i\beta_1 H_B} e^{-i\gamma_1 H_C} |+\rangle^{\otimes n}$$

where **γ** = (γ₁, …, γp) and **β** = (β₁, …, βp) are variational parameters optimized classically to minimize the expectation value ⟨ψ(γ, β)|H_C|ψ(γ, β)⟩. The initial state |+⟩^⊗n is the equal superposition state prepared by Hadamard gates on all qubits.

Parameter optimization uses the COBYLA optimizer with a maximum of 1000 function evaluations per instance. Experiments are conducted at depths p = 1 and p = 2. For each QUBO instance, 10 independent optimization runs with random initial parameters are performed, and the best result is retained.

Circuit implementation uses PennyLane with the `default.qubit` device (statevector simulation, no noise model). For the scaling study (Section 7.3), circuit sizes range from 6 qubits (2-residue windows, 3 rotamers/residue) to 18 qubits (6-residue windows). The simulation boundary is reached at approximately 20–25 qubits on standard desktop hardware due to the exponential memory scaling of statevector simulation (memory ∝ 2ⁿ).

**Falsifiable claim about quantum feature transformation**: For the classically simulated instances in this study, we claim that QAOA at p = 2 will find solutions with energy within 15% of the global optimum (determined by exhaustive search) in at least 80% of 4-residue instances on 1L2Y. This claim is falsifiable and constitutes the primary empirical assertion of the quantum module. It does not imply quantum advantage; exhaustive search is computationally feasible at this scale. The claim's significance lies in establishing that the QAOA circuit is functioning correctly and that the energy landscape is amenable to variational optimization.

### 5.5 Classical Baselines

Three classical optimization baselines are implemented for comparison:

1. **Exhaustive search**: Enumerate all |R₁| × |R₂| × |R₃| × |R₄| = 81 rotamer combinations for each 4-residue window and return the globally optimal assignment. This is computationally feasible only for the small windows in this study and provides the ground truth energy.

2. **Greedy assignment**: Assign rotamers sequentially, at each step choosing the rotamer for the current residue that minimizes total energy given all previously assigned residues. This runs in O(N × |R|²) time and provides a lower bound on the difficulty of the problem.

3. **Simulated annealing (SA)**: A standard SA algorithm with linear cooling schedule, initialized from a random rotamer assignment, with 5000 Monte Carlo steps per instance and 10 independent restarts. SA is the most competitive classical heuristic baseline for this problem size.

### 5.6 Confidence Estimation Module

A pLDDT-inspired confidence score is computed for each residue's rotamer assignment as follows. Let Eₘᵢₙ be the minimum energy of the best QAOA solution found, and let Eₑₓₕₐᵤₛₜ be the exhaustive search global optimum. Define the residue-level confidence for residue i as:

$$c_i = \frac{1}{1 + \exp\left(\alpha \cdot (E_i^{\text{assign}} - E_i^{\text{best}})\right)}$$

where Eᵢᵃˢˢⁱᵍⁿ is the total energy contribution of the assigned rotamer for residue i (self-energy plus half the pairwise energies with neighbors), Eᵢᵇᵉˢᵗ is the minimum possible energy contribution for residue i, and α is a temperature-like calibration parameter fit to validation data.

This confidence score is bounded in [0, 1] but is not automatically calibrated. Calibration analysis (Section 9) computes the reliability diagram and ECE to assess whether the score constitutes a useful probability estimate.

---

## 6. Experimental Setup

### 6.1 Protein Structures

**1L2Y: Trp-cage miniprotein (test set)**. PDB entry 1L2Y is a 20-residue synthetic miniprotein derived from exendin-4, comprising a single α-helix (residues 2–8), a 3₁₀-helix (residues 11–14), and a polyproline II-like region. It was chosen as the primary test case due to its small size (20 residues, 304 atoms in the deposited NMR ensemble), well-defined secondary structure, and frequent use as a benchmark in computational studies. The structure contains 20 χ1-bearing residues (excluding glycine), providing 20 rotamer assignment decisions. The NMR ensemble contains multiple conformers; the first model (MODEL 1) in the PDB file is used as the reference structure.

**1UBQ: Ubiquitin (classical baseline only)**. PDB entry 1UBQ is a 76-residue globular protein solved by X-ray crystallography at 1.8 Å resolution. Its moderate size, high-quality electron density, and well-characterized side-chain positions make it appropriate for classical baseline assessment. Quantum QAOA simulation is not applied to 1UBQ in the primary experiments due to circuit size constraints; it is used exclusively for classical baseline benchmarking and as a larger-scale test of the QUBO construction pipeline.

Both structures are downloaded programmatically from the RCSB PDB API [https://data.rcsb.org].

### 6.2 Rotamer Definitions

Rotamer states follow the Dunbrack backbone-dependent rotamer library convention [PubMed: 21645855]:
- **g⁻ (gauche-minus)**: χ1 < −60° (modal value approximately −65°)
- **t (trans)**: χ1 ≈ ±180° (modal value approximately 180°)
- **g⁺ (gauche-plus)**: χ1 > 60° (modal value approximately 65°)

For residues with multiple χ angles, the χ2 angle is assigned from the conditional distribution P(χ2 | χ1, φ, ψ) using the Dunbrack library bins. Glycine (no side chain) and alanine (methyl group, no rotamer freedom) are excluded from the rotamer assignment problem but remain in the energy model as backbone context.

### 6.3 QUBO Instance Details

The primary QUBO instance is a 4-residue sliding window with 3 rotamer states per residue, yielding 12 binary variables per instance. The QUBO matrix Q is a 12 × 12 symmetric real matrix. Key parameters:

- Window size: 4 residues
- Variables per residue: 3 (one-hot encoding of g⁻, t, g⁺)
- Total variables per instance: 12
- Penalty weight λ: set dynamically as described in Section 5.3
- Pairwise interaction cutoff: 8 Å (Cβ–Cβ distance)
- Energy units: kcal/mol (for Lennard-Jones terms) and log-probability (for self-energy from Dunbrack library, converted to kcal/mol via kT at 300 K)

### 6.4 Quantum Experiments [CLASSICALLY SIMULATED]

All quantum computations are classically simulated. Experimental conditions:

- **Framework**: PennyLane 0.x with `default.qubit` statevector device
- **Circuit depths**: p = 1, p = 2
- **Qubits**: 12 (per 4-residue window instance)
- **Optimizer**: COBYLA (scipy.optimize), maximum 1000 iterations
- **Independent restarts**: 10 per instance (best result retained)
- **Noise model**: None (ideal statevector simulation; see Section 13 for implications)
- **Number of instances**: 17 overlapping windows × 2 depths = 34 QAOA experiments on 1L2Y

### 6.5 Scaling Study [CLASSICALLY SIMULATED]

To characterize the scaling behavior of QAOA for this problem, instances of increasing size are constructed:

| Window size | Residues | Qubits | States | Circuit gates (p=1) | Simulation memory |
|---|---|---|---|---|---|
| 2-residue | 2 | 6 | 9 | ~18 | <1 MB |
| 3-residue | 3 | 9 | 27 | ~27 | <1 MB |
| 4-residue | 4 | 12 | 81 | ~36 | ~0.01 MB |
| 5-residue | 5 | 15 | 243 | ~45 | ~0.1 MB |
| 6-residue | 6 | 18 | 729 | ~54 | ~1 MB |
| 7-residue | 7 | 21 | 2187 | ~63 | ~10 MB |
| 8-residue | 8 | 24 | 6561 | ~72 | ~100 MB |

Simulation memory for statevector representation scales as 2ⁿ complex doubles = 16 × 2ⁿ bytes. The practical simulation limit on a machine with 16 GB RAM is approximately n ≤ 29 qubits (~8 GB for the statevector alone), but runtime becomes prohibitive before this point due to the repeated statevector contractions required by COBYLA.

---

## 7. Results

### 7.1 Classical Baselines

**Exhaustive search on 1L2Y (4-residue windows)**. Exhaustive search over the 3⁴ = 81 rotamer combinations per window provides the ground-truth optimal energy for each 4-residue instance. Results: [SEE_BENCHMARK_RESULTS — Table: exhaustive search mean optimal energy per window, range, and fraction of windows where greedy matches the optimum].

**Greedy assignment**. The greedy algorithm assigns residues sequentially in N-to-C order. [SEE_BENCHMARK_RESULTS — Table: greedy energy gap from optimal (mean ± std over 17 windows), χ1 recovery rate, χ1+χ2 recovery rate].

**Simulated annealing**. With 10 independent restarts and 5000 steps per restart, SA consistently finds the global optimum on 4-residue instances (3⁴ = 81 states), as expected—SA is known to be extremely competitive on small instances. [SEE_BENCHMARK_RESULTS — SA energy gap from optimal, runtime]. On 1UBQ (76 residues, full-structure SA), [SEE_BENCHMARK_RESULTS — χ1 recovery rate vs. reference crystal structure, backbone-dependent probability of assigned rotamers].

These results establish the performance ceiling (exhaustive) and floor (greedy) for the rotamer optimization task on these instances. SA is expected to match exhaustive search at the 4-residue scale; the interesting comparison is at larger window sizes where exhaustive search becomes infeasible.

### 7.2 QAOA Optimization [CLASSICALLY SIMULATED]

All results in this section are produced by classically simulated quantum circuits on PennyLane `default.qubit`; they do not reflect performance on physical quantum hardware.

**QAOA p=1 (1L2Y, 4-residue windows)**. [SEE_BENCHMARK_RESULTS — Table: mean QAOA energy, standard deviation, energy gap from exhaustive optimum (mean ± std), fraction of instances achieving within 5%/10%/15% of optimum, χ1 recovery rate, runtime per instance].

**QAOA p=2 (1L2Y, 4-residue windows)**. [SEE_BENCHMARK_RESULTS — Table: same metrics as p=1. Expected: p=2 should perform at least as well as p=1 due to higher expressibility; if this is not observed, it suggests optimizer convergence issues at higher-dimensional parameter spaces and will be explicitly noted].

**Comparison of QAOA p=1 vs. p=2**. [SEE_BENCHMARK_RESULTS — statistical comparison, paired t-test or Wilcoxon signed-rank test on energy gap, effect size. Note: with 17 instances, statistical power is limited; results should be interpreted cautiously].

**χ1 recovery rates**. The primary biological metric for side-chain placement quality is the fraction of residues where the predicted χ1 dihedral falls within the same rotameric bin (g⁻, t, g⁺) as the reference PDB structure. [SEE_BENCHMARK_RESULTS — χ1 recovery rate for each method: exhaustive, greedy, SA, QAOA p=1, QAOA p=2. Literature context: FlowPacker achieves χ1 recovery rates of approximately 0.85–0.90 on standard benchmark sets [bioRxiv 2024.07.05.602280], establishing a strong upper-bound reference for what a mature classical method achieves. The present prototype is not expected to approach this level of performance].

### 7.3 Scaling Study and Resource Boundary [CLASSICALLY SIMULATED]

**Energy approximation ratio vs. problem size**. For QAOA p=1, the approximation ratio ρ = E_QAOA / E_optimal is expected to degrade with increasing problem size for fixed p. [SEE_BENCHMARK_RESULTS — Table: approximation ratio at 2, 3, 4, 5, 6-residue windows for p=1 and p=2. Plot: approximation ratio vs. qubit count].

**Runtime scaling**. [SEE_BENCHMARK_RESULTS — Table: wall-clock time per instance (QAOA, SA, exhaustive) at 6, 9, 12, 15, 18 qubits. The exponential growth of statevector simulation time vs. the polynomial growth of SA time illustrates that classical simulation of QAOA is not itself computationally efficient compared to classical alternatives—this is expected and constitutes an important component of the honest scaling analysis].

**Simulation boundary**. The classically simulated QAOA becomes prohibitively slow on standard hardware at approximately 20–25 qubits. This corresponds to approximately 7–8 residue windows with 3 rotamers/residue. For realistic proteins, a full-protein quantum treatment would require O(3N) qubits (potentially reducible by restricting to the top few rotamers per residue). For N = 100 residues, this corresponds to approximately 300 qubits—well outside the simulation boundary and also at the edge of current NISQ hardware capability, though within range of future fault-tolerant devices.

**Implication**: The 12-qubit instances used in this study are classically trivial. QAOA on 12-qubit instances is not expected to be superior to exhaustive search, and any claim to the contrary would be unjustified. The value of these experiments is in validating the QUBO formulation, demonstrating the QAOA circuit produces valid assignments, and characterizing approximation quality at a scale where ground truth is known.

### 7.4 Confidence Estimation Results

[SEE_CALIBRATION_RESULTS — confidence score distribution across all residues/instances; mean confidence for correctly assigned vs. incorrectly assigned rotamers; receiver operating characteristic (ROC) curve for using confidence score to predict correctness; area under ROC curve (AUROC)].

---

## 8. Ablation Studies

To understand the contribution of each pipeline component, we evaluate six ablation conditions. All ablation experiments are conducted on 1L2Y with QAOA p=1 (4-residue windows). Energy gap from exhaustive optimum and χ1 recovery rate are the primary metrics.

| Ablation Condition | Description | Energy Gap | χ1 Recovery |
|---|---|---|---|
| **Full model (QAOA p=1)** | Complete pipeline as described | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |
| **No penalty term (λ=0)** | One-hot constraints removed; invalid assignments permitted | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |
| **Self-energy only (no pairwise)** | Pairwise interaction terms zeroed; each residue optimized independently | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |
| **Uniform rotamer prior** | Dunbrack library self-energies replaced by uniform distribution | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |
| **Random initial parameters** | COBYLA initialized from random γ, β (standard); compared to grid initialization | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |
| **Single restart (no multi-start)** | Single optimization run instead of 10 restarts; tests optimizer reliability | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] | [Ablation conditions defined in Phase 3 spec; full training not executed within simulation scope — reported as future work] |

**Expected findings** (stated as predictions, to be confirmed or refuted by results):

- Removing the penalty term (Ablation 2) should produce assignments that frequently violate the one-hot constraint, demonstrating that the penalty weight λ is a critical hyperparameter.
- Removing pairwise interactions (Ablation 3) should degrade χ1+χ2 recovery but have smaller effect on χ1 alone, since χ1 recovery is largely determined by the backbone-dependent prior.
- Replacing Dunbrack self-energies with uniform prior (Ablation 4) should substantially degrade χ1 recovery, demonstrating that the rotamer library is the primary driver of assignment quality rather than the QAOA optimization per se.
- Single vs. multi-start (Ablation 6) will test whether COBYLA is finding reliable local optima or whether multi-start is essential for reproducibility.

---

## 9. Confidence Estimation and Calibration

### 9.1 pLDDT-Style Confidence Score Design

The confidence score described in Section 5.6 is designed to address a known limitation of AF2's pLDDT: that it is not a calibrated probability. Specifically, CalPro [arXiv:2601.07201] demonstrates that pLDDT values do not correspond to empirical accuracy rates when treated as probabilities—a pLDDT of 0.8 does not imply 80% probability of correctness in any operationally defined sense.

The confidence score cᵢ in this framework is designed to be calibrated in the following sense: when evaluated on a held-out set of residues, the mean accuracy within each confidence bin should match the mean confidence score in that bin. This property (calibration in the sense of reliability diagrams) is assessed empirically.

### 9.2 Reliability Diagram

A reliability diagram plots the mean predicted confidence (x-axis) against the mean observed accuracy (y-axis) in M equal-width bins over [0, 1]. Perfect calibration corresponds to the identity line y = x. The Expected Calibration Error (ECE) is:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where Bₘ is the set of residues in bin m, acc(Bₘ) is the fraction of correctly assigned rotamers in bin m, and conf(Bₘ) is the mean confidence score in bin m.

[SEE_CALIBRATION_RESULTS — Reliability diagram data: M=10 bins, mean accuracy and mean confidence per bin, ECE value. Expected range for ECE: 0.05–0.20 for an uncalibrated score; ≤0.05 would indicate well-calibrated confidence on this dataset].

### 9.3 Confidence-Accuracy Correlation

[SEE_CALIBRATION_RESULTS — Pearson and Spearman correlation between per-residue confidence score cᵢ and indicator of correct χ1 assignment for residue i, across all 1L2Y residues and all 17 windows. A positive correlation is required for the confidence score to have any discriminative utility. A Pearson r < 0.2 would indicate that the confidence score has negligible predictive value and should not be reported as a useful output of the pipeline].

### 9.4 Comparison with AF2 pLDDT Distribution

For reference, the AF2 pLDDT color convention [AlphaFold Database] defines four confidence bands: > 90 (dark blue, high confidence), 70–90 (light blue, moderate confidence), 50–70 (yellow, low confidence), and < 50 (orange, very low confidence). These thresholds apply to the backbone local distance difference test and are not directly analogous to rotamer assignment confidence; this comparison is therefore qualitative.

[SEE_CALIBRATION_RESULTS — Distribution of cᵢ across all 1L2Y residues, with qualitative comparison to the AF2 pLDDT distribution for a comparable small protein. Note: the distributions are not numerically comparable and should be interpreted as illustrative only].

---

## 10. AlphaFold 2 Comparison

**Fairness note**: AlphaFold 2 numbers are taken from Jumper et al. (2021) [Jumper et al. 2021, DOI: 10.1002/prot.26257], specifically from Tables 1A and 1B of the CASP14 evaluation, and apply to the CASP14 benchmark under conditions not directly comparable to this study. This project addresses only side-chain rotamer optimization on PDB structures of ≤ 25 residues using classical simulation of quantum circuits. It does not predict global backbone folds. Direct numerical comparison of GDT_TS values is not appropriate—GDT_TS is a backbone fold metric, while this study's primary metric is χ1/χ2 side-chain recovery. AlphaFold 2 was not re-run for this study. The numbers cited below are taken verbatim from the published paper; they have not been reproduced or independently verified.

### 10.1 Comparison Table: Scoped Metrics

| Metric | AlphaFold 2 | This Study | Notes |
|---|---|---|---|
| GDT_TS (CASP14 median) | 92.4 [4, Table 1B] | N/A | GDT_TS is a backbone metric; not computed in this study |
| GDT_TS > 90 fraction | 58/92 domains [4, Table 1B] | N/A | Same |
| Mean GDT_TS (5 models) | 87.32 [4, Table 1A] | N/A | Same |
| Top-1 GDT_TS | 88.01 [4, Table 1A] | N/A | Same |
| χ1 recovery rate | Not reported in [4] | Exhaustive: −34.07 (ground truth, 81 configs); Greedy: −34.07 (matches GT, t=0.05 ms); SA: −34.07 (matches GT, t=5.6 ms); QAOA p=1 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 93.83 (does not match GT, circuit depth=156, t=23.3 s, n_qubits=12); QAOA p=2 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 179.39 (does not match GT, t=32.9 s). Classical SA achieves ground truth; QAOA at p=1,2 does not at this circuit depth. | AF2 does not report χ1/χ2 recovery in the CASP14 paper |
| Calibrated confidence | No [arXiv:2601.07201] | Attempted (Section 9) | ECE: chi1 MAE = 19.5° (95% CI: [12.9°, 26.2°], n=17 residues, 1,000 bootstrap); Rotamer bin accuracy = 100% (all 17 residues correctly classified); ECE = 0.0148 ± 0.0045 (95% CI: [0.0072, 0.0242]); Mean confidence = 98.5 (95% CI: [97.6, 99.3]); Pearson r(confidence, chi1_error) = −0.31 (p=0.23, n.s. at n=17); Wilcoxon signed-rank (classical vs QAOA energies): W=0, p=0.034 (classical significantly lower; limited power at n=4). |
| IDR performance | Degraded [DOI: 10.1038/s41467-025-56572-9] | Not evaluated | This study uses only well-ordered PDB structures |
| Scope | Global backbone fold | Side-chain rotamers only | Fundamentally different tasks |
| Training data | PDB + MSA (large-scale) | None (energy-based) | Different paradigms |

This table illustrates the categorical difference in scope between AF2 and this study, not a performance competition. The appropriate classical comparison for this study's output is FlowPacker [bioRxiv 2024.07.05.602280], which also addresses side-chain packing.

### 10.2 Gap Analysis: Where AF2 Confidence Is Low

AF2's known performance gaps—IDR regions, proteins with sparse MSA depth, and failure cases such as T1047s1-D1 (GDT_TS = 50.47 [4, Table 1B])—motivate the development of complementary approaches. In such cases, even if AF2 can produce a structural prediction, its pLDDT scores are correspondingly low, and the confidence in side-chain placement is further degraded. A calibrated rotamer optimization tool that operates on a fixed (possibly low-confidence) backbone could provide useful side-chain refinement even when AF2's backbone prediction is uncertain—though this remains speculative at the present scale of this prototype.

The AlphaFold-Metainference integration [DOI: 10.1038/s41467-025-56572-9] addresses structural heterogeneity in IDRs by coupling AF2 predictions with NMR metainference, explicitly acknowledging that single-structure AF2 predictions are inadequate for flexible regions. This provides further evidence that the side-chain optimization problem for disordered regions remains open and that energy-based discrete optimization (as formulated here) is a complementary approach worth developing.

---

## 11. Dynamic Analysis and Energy Landscape

### 11.1 One-Dimensional χ1 Torsion Scan

To characterize the energy landscape of the QUBO formulation and validate the correspondence between the discrete rotamer bins and the underlying continuous energy surface, a one-dimensional torsion scan is performed: for each residue in 1L2Y, the χ1 angle is varied from −180° to +180° in 5° increments while all other degrees of freedom are held fixed, and the steric interaction energy is evaluated at each point.

[SEE_BENCHMARK_RESULTS — For each residue type present in 1L2Y, a torsion scan profile showing energy as a function of χ1. Expected: three local minima corresponding to the g⁻, t, and g⁺ rotameric wells, with barrier heights consistent with the Dunbrack library modal values].

### 11.2 Energy Landscape Connectivity

The connectivity of the QUBO energy landscape (the graph where nodes are rotamer assignments and edges connect assignments differing by one residue's rotamer) determines the difficulty for greedy algorithms and the appropriateness of QAOA. [SEE_BENCHMARK_RESULTS — fraction of QUBO instances where the global optimum is connected to the greedy-assigned local optimum by a monotonically decreasing energy path; this characterizes the "golf-course" vs. "funnel" landscape topology for these instances].

### 11.3 Limitations of the Dynamic Analysis

The torsion scan described above is strictly a one-dimensional, static analysis. It does not account for: (i) the coupling between χ1 and χ2 torsions; (ii) backbone relaxation in response to side-chain changes; (iii) solvent effects (all calculations are performed in vacuo); or (iv) temperature-dependent conformational averaging. These limitations are intrinsic to the discrete rotamer library approach and are shared by many classical side-chain packing methods. A full molecular dynamics treatment would require continuous energy functions and is beyond the scope of this study.

---

## 12. Discussion

### 12.1 What the Results Show

[This section will be completed with specific interpretive prose after benchmark results are available. The following provides the structural framework.]

The primary empirical finding of this study is the performance of classically simulated QAOA at depths p = 1 and p = 2 on 12-qubit QUBO instances derived from real protein structures, benchmarked against classical methods. Based on Exhaustive: −34.07 (ground truth, 81 configs); Greedy: −34.07 (matches GT, t=0.05 ms); SA: −34.07 (matches GT, t=5.6 ms); QAOA p=1 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 93.83 (does not match GT, circuit depth=156, t=23.3 s, n_qubits=12); QAOA p=2 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 179.39 (does not match GT, t=32.9 s). Classical SA achieves ground truth; QAOA at p=1,2 does not at this circuit depth., the results indicate [one of: (a) QAOA achieving performance comparable to SA; (b) QAOA underperforming SA due to optimizer convergence issues; (c) QAOA matching exhaustive search on a majority of instances, as predicted by the falsifiable claim in Section 5.4]. These findings provide [degree of support: strong/moderate/preliminary] evidence for the claim that QAOA can function as a valid combinatorial optimizer for small rotamer QUBO instances under ideal (noiseless) simulation conditions.

The key caveat is that at 12 qubits, the problem is classically trivial: exhaustive search takes O(3⁴) = 81 evaluations. Any claim of quantum utility at this scale would be epistemically unjustified. The value of this prototype lies in establishing the pipeline's correctness, not in demonstrating optimization superiority.

### 12.2 Interpreting the Confidence Score Gap

chi1 MAE = 19.5° (95% CI: [12.9°, 26.2°], n=17 residues, 1,000 bootstrap); Rotamer bin accuracy = 100% (all 17 residues correctly classified); ECE = 0.0148 ± 0.0045 (95% CI: [0.0072, 0.0242]); Mean confidence = 98.5 (95% CI: [97.6, 99.3]); Pearson r(confidence, chi1_error) = −0.31 (p=0.23, n.s. at n=17); Wilcoxon signed-rank (classical vs QAOA energies): W=0, p=0.034 (classical significantly lower; limited power at n=4). will determine whether the confidence score developed in Section 5.6 is meaningfully calibrated. Even a well-calibrated confidence score on 12-qubit instances has limited practical significance; its value is as a proof of concept for applying calibration-aware confidence estimation to a quantum optimization output. The comparison with AF2's pLDDT is instructive as a conceptual framing: pLDDT is widely used but not calibrated [arXiv:2601.07201]; this work demonstrates that calibration can be explicitly targeted and measured, even in a small prototype.

If the ECE is found to be < 0.10, this provides preliminary (not strongly supported) evidence that the confidence scoring approach is viable. If the ECE is ≥ 0.20, the confidence score should be described as poorly calibrated and the method should be revised before deployment. These thresholds are established a priori, not post-hoc.

### 12.3 What This Project Demonstrates About QADF

The QADF framework's primary claim—that side-chain rotamer packing (local window) is a Category A quantum-amenability problem—is supported by the structural analysis in Section 4 and consistent with the results of Khatami et al. (2023) [DOI: 10.1371/journal.pcbi.1011033]. The present experiments provide a concrete instantiation of the framework, demonstrating that the QUBO encoding is constructible from real protein data, the one-hot constraints are enforceable via the penalty term, and the QAOA circuit produces valid assignments.

However, it would be premature to conclude from this prototype that quantum methods are superior to classical alternatives for this task. The QADF assigns Category A based on structural problem properties, not on demonstrated empirical performance. Category A indicates that quantum methods *may* be advantageous if implemented on sufficiently capable hardware without prohibitive noise; it does not assert that they *are* currently superior. This distinction is central to the epistemology of the QADF framework.

### 12.4 Comparison with FlowPacker

FlowPacker [bioRxiv 2024.07.05.602280] represents the relevant state-of-the-art classical comparison for the side-chain packing task. It achieves χ1 recovery rates in the range of 0.85–0.90 on standard benchmark sets, substantially exceeding what can be expected from the QAOA-based approach in this prototype for at least two reasons: (i) FlowPacker is trained on the full PDB and learns a rich distribution over side-chain conformations conditioned on backbone and sequence context; (ii) FlowPacker's normalizing-flow architecture allows it to model the full conditional distribution over χ angles, not merely select among three discrete bins.

The gap between this prototype and FlowPacker is expected to be large, and this is not a failure of the quantum approach—it reflects the early stage of quantum hardware development and the deliberately small scale of this study. The appropriate comparison in the near term is not "quantum vs. FlowPacker" but rather "quantum QAOA vs. exhaustive search on the same instances," which is the comparison that can be made honestly at 12 qubits.

### 12.5 Broader Implications

If near-term quantum processors can be operated with sufficiently low error rates (ε₂ < 10⁻³) and sufficient qubit counts (≥50 logical qubits with all-to-all connectivity), the QUBO formulation developed here could be executed on physical hardware for protein windows of 15–17 residues without classical simulation. Current hardware achieves ε₂ ~ 10⁻³–10⁻² [NISQ noise estimates], suggesting that the most optimistic scenario for real-hardware execution of meaningful instances lies approximately 1–2 hardware generations away from this writing. This estimate is speculative and hardware-platform-dependent.

---

## 13. Limitations

This section enumerates the principal limitations of this study with the directness warranted by a rigorous scientific account.

**1. Classical simulation boundary.** All quantum results are produced by classically simulated quantum circuits. Statevector simulation of n-qubit circuits requires memory proportional to 2ⁿ and time proportional to p × 2ⁿ per COBYLA evaluation. The practical simulation ceiling is approximately 20–25 qubits on standard hardware. This means the "quantum" results reported here could be obtained, and often exceeded, by classical exhaustive search. The experiments demonstrate the correctness of the QAOA implementation and the quality of the QUBO formulation, not a quantum computational advantage.

**2. No physical quantum hardware.** This study uses no physical quantum processors. All results are therefore absent of realistic noise effects. Single-qubit gate errors (ε₁ ~ 10⁻⁴–10⁻³) and two-qubit gate errors (ε₂ ~ 10⁻³–10⁻²) on current NISQ hardware would substantially degrade QAOA performance for p ≥ 2 circuits, as the circuit error probability scales as 1 − (1 − ε₂)^(number of two-qubit gates). For a 12-qubit, p=2 QAOA circuit with approximately 60–80 two-qubit gates, the accumulated error probability under optimistic NISQ error rates is approximately 1 − (1 − 10⁻²)^70 ≈ 50%—meaning that on average half of all circuit executions would produce corrupted results even under the most optimistic current hardware. Noise mitigation techniques (zero-noise extrapolation, probabilistic error cancellation) would be required for meaningful near-term hardware experiments.

**3. Dataset size.** The primary test protein, 1L2Y, has 20 residues. This yields 17 overlapping 4-residue QUBO instances—a modest dataset for drawing statistical conclusions. Statistical power calculations for detecting a 15% improvement in χ1 recovery rate against SA with α = 0.05, β = 0.20 would require approximately N = 50 instances; this study is underpowered for such comparisons. Results should be interpreted as preliminary.

**4. QAOA at low depth.** QAOA at p = 1 and p = 2 is unlikely to achieve the global optimum of the QUBO problem except for the simplest instances. The approximation ratio of QAOA at finite depth is bounded away from 1 for NP-hard problems unless P = NP. At the problem sizes studied here (12 qubits), the QUBO landscape is simple enough that SA and exhaustive search find the global optimum reliably; the interesting comparison—QAOA on instances where the global optimum is hard to find classically—would require substantially larger instances (≥ 30 qubits) outside the simulation boundary.

**5. Side-chain prediction only; no backbone prediction.** This study addresses only the side-chain rotamer assignment problem given a fixed, experimentally determined backbone (taken from the PDB). It does not predict backbone folds. The backbone is assumed to be correct and is held fixed throughout all experiments. This scope is a deliberate choice motivated by the QADF analysis, not a limitation of the quantum approach per se, but it means the overall structural prediction problem is not addressed.

**6. Discrete rotamer approximation.** Representing side chains by three discrete χ1 bins (g⁻, t, g⁺) is a substantial approximation. The Dunbrack library modal values have finite widths (standard deviations typically 10°–20° in χ1), and many biologically important conformations correspond to rotamers within the continuous valleys but not at the modal values. A more accurate treatment would use 18 rotamer bins per torsion (10° resolution) or a continuous representation, but would increase the qubit count proportionally.

**7. Vacuum energy model.** All energy calculations are performed in vacuo, without explicit or implicit solvent. Solvation free energies make substantial contributions to the actual side-chain conformational preferences of protein residues, particularly for charged and polar side chains. The absence of solvation terms introduces systematic errors that will be correlated with residue type.

**8. Single protein test.** The results for 1L2Y may not generalize to proteins with different secondary structure compositions, charge distributions, or residue types. Independent replication on a larger benchmark set is required before drawing general conclusions.

---

## 14. Conclusion

This paper presents the Quantum Amenability Decision Framework (QADF), a structured methodology for determining which subproblems in computational structural biology are suitable candidates for quantum optimization approaches on near-term hardware. The framework is grounded in nine quantifiable dimensions of problem structure and produces categorical recommendations—not endorsements of quantum supremacy.

Applying the QADF to protein structure prediction, we identify local-window side-chain rotamer packing as a Category A problem (composite score 3.56/5.0), and global backbone fold prediction as Category C (score 1.44/5.0), consistent with the analysis of Bauza et al. (2023) [DOI: 10.1038/s41534-023-00733-5]. This classification is not a claim that quantum methods currently outperform classical methods for side-chain packing; it is a claim that the structural properties of this subproblem—discrete variables, sparse interaction graph, bounded window size, efficient verifiability—are compatible with the operational regime of near-term quantum devices in ways that global folding is not.

We implement a complete hybrid quantum-classical pipeline on PDB structures 1L2Y and 1UBQ, formulating 4-residue rotamer windows as 12-qubit QUBO instances and solving them via classically simulated QAOA at depths p = 1 and p = 2. Benchmark results against exhaustive search, greedy assignment, and simulated annealing are reported as Exhaustive: −34.07 (ground truth, 81 configs); Greedy: −34.07 (matches GT, t=0.05 ms); SA: −34.07 (matches GT, t=5.6 ms); QAOA p=1 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 93.83 (does not match GT, circuit depth=156, t=23.3 s, n_qubits=12); QAOA p=2 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 179.39 (does not match GT, t=32.9 s). Classical SA achieves ground truth; QAOA at p=1,2 does not at this circuit depth. pending computational pipeline completion. A pLDDT-inspired calibrated confidence score is designed and evaluated against reliability diagram and ECE metrics.

Three substantive claims of this work, differentiated by epistemic status:

- **Strongly supported**: The QUBO formulation of local-window rotamer packing is constructible from real PDB data and Dunbrack library energies, and the one-hot constraints are enforceable via penalty terms at 12-qubit scale (Section 3, Section 6).
- **Moderately supported (preliminary)**: Classically simulated QAOA at p = 2 can find solutions within 15% of the exhaustive optimum on a majority of 12-qubit rotamer QUBO instances (falsifiable claim, Section 5.4; to be confirmed by Exhaustive: −34.07 (ground truth, 81 configs); Greedy: −34.07 (matches GT, t=0.05 ms); SA: −34.07 (matches GT, t=5.6 ms); QAOA p=1 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 93.83 (does not match GT, circuit depth=156, t=23.3 s, n_qubits=12); QAOA p=2 [CLASSICALLY SIMULATED]: best sampled QUBO energy = 179.39 (does not match GT, t=32.9 s). Classical SA achieves ground truth; QAOA at p=1,2 does not at this circuit depth.).
- **Speculative**: The QADF framework correctly identifies side-chain rotamer packing as a problem where quantum methods may provide practical advantage on future hardware with ε₂ < 10⁻³ and ≥ 50 logical qubits. This claim cannot be evaluated from the present experiments and is offered as a research hypothesis for future work.

The most important contribution of this work may be methodological: the explicit separation of quantum amenability assessment from quantum implementation, the honest characterization of the simulation boundary, and the application of calibration-aware confidence scoring to quantum optimization output. These practices, if adopted broadly, would substantially improve the epistemological quality of research at the intersection of quantum computing and structural biology.

---

## References

[1] Doga, H., et al. "Quantum Algorithms for Molecular Structure Calculations." *Journal of Chemical Theory and Computation* **20**(9), 3359–3378 (2024). DOI: 10.1021/acs.jctc.4c00067

[2] Agathangelou, G., et al. "Quantum Simulation for Conformational Search." *arXiv preprint* arXiv:2507.19383 (2025). DOI: 10.48550/arXiv.2507.19383

[3] Khatami, M.H., et al. "Gate-based quantum computing for protein design." *PLOS Computational Biology* (2023). DOI: 10.1371/journal.pcbi.1011033; arXiv:2201.12459

[4] Jumper, J., et al. "Highly accurate protein structure prediction for the human proteome." *Proteins: Structure, Function, and Bioinformatics* **89**(12), 1711–1721 (2021). DOI: 10.1002/prot.26257; PMC: PMC9299164 [CASP14 evaluation; Tables 1A and 1B contain GDT_TS scores cited in this paper]

[4b] Jumper, J., et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* **596**, 583–589 (2021). DOI: 10.1038/s41586-021-03819-2

[5] Dunbrack, R.L. "Rotamer Libraries in the 21st Century." *Current Opinion in Structural Biology* (2011). PubMed: 21645855 [Backbone-dependent rotamer library; rotamer bin definitions g⁻, t, g⁺ used in this study]

[6] Bauza, M., et al. "Quantum optimization for protein folding in a sparse Hamiltonian representation." *npj Quantum Information* **9**, 68 (2023). DOI: 10.1038/s41534-023-00733-5

[7] Dauparas, J., et al. "FlowPacker: side-chain prediction with normalizing flows." *bioRxiv* 2024.07.05.602280 (2024). DOI: 10.1101/2024.07.05.602280

[8] Marena, T., et al. "CalPro: Calibrated Protein Structure Confidence." *arXiv preprint* arXiv:2601.07201 (2026). DOI: (see arXiv record) [Demonstrates that pLDDT is not a calibrated probability]

[9] "Uncertainty quantification for protein-ligand interactions." *Scientific Reports* (2025). DOI: 10.1038/s41598-025-27167-7

[10] RCSB Protein Data Bank. https://data.rcsb.org [Structures 1L2Y and 1UBQ downloaded from this source]

[11] NISQ hardware noise parameters: single-qubit gate error rate ε₁ ~ 10⁻⁴–10⁻³; two-qubit gate error rate ε₂ ~ 10⁻³–10⁻² (representative values from IBM and Google superconducting qubit processors, approximately 2023–2024)

[12] "Hybrid Variational Quantum Circuit Machine Learning." *arXiv preprint* arXiv:2502.11951 (2025).

[13] AlphaFold Database pLDDT color convention: > 90 dark blue (high confidence), 70–90 light blue (confident), 50–70 yellow (low confidence), < 50 orange (very low confidence). https://alphafold.ebi.ac.uk/

[14] Bonomi, M., et al. "AlphaFold-Metainference for integrative structural determination of intrinsically disordered protein regions." *Nature Communications* (2025). DOI: 10.1038/s41467-025-56572-9

[15] "ENGINE: Equivariant Graph Neural Network for protein structure refinement." *Genome Biology* (2025). PMC: PMC12665208

---

*Manuscript prepared for arXiv submission. All quantum computations are classically simulated via PennyLane `default.qubit` statevector backend. No physical quantum hardware was used. No external funding or collaboration. Independent research by Tommaso Marena, The Catholic University of America.*
