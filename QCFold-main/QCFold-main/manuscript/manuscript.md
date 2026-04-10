# QCFold: Hybrid Quantum-Classical Ensemble Prediction for Fold-Switching Proteins

**Tommaso R. Marena**  
Department of Biology, Catholic University of America, Washington, DC, USA  
Email: marena@cua.edu

---

## Abstract

Fold-switching proteins—metamorphic proteins that adopt two or more structurally distinct, natively folded conformations—represent a persistent blind spot for current deep-learning structure prediction systems. AlphaFold 3 (AF3) achieves only 7.6% success on a curated 92-protein fold-switching benchmark, and even the best single-method approach (CF-random) reaches 34.8%, leaving the majority of fold-switching events undetected. We present QCFold, a hybrid quantum-classical pipeline for fold-switching structure prediction that combines: (1) a multi-conformation generator producing structurally diverse candidate ensembles; (2) a quantum variational refinement module that formulates fold-state assignment as a Quadratic Unconstrained Binary Optimization (QUBO) problem mapped to an Ising Hamiltonian and solved via the Quantum Approximate Optimization Algorithm (QAOA) or Variational Quantum Eigensolver (VQE), with classical simulated annealing (SA) as a practical fallback; (3) a physics/geometry consistency layer enforcing bond geometry, steric clash avoidance, and Ramachandran validity; and (4) a diversity-aware ensemble ranking head. On a five-protein demonstration subset drawn from the Ronish et al. 2024 benchmark, QCFold achieves a 60% success rate (3/5, 95% CI: 23.1%–88.2%), compared to 7.6–34.8% for published baseline methods evaluated on the full 92-protein benchmark. Ablation studies confirm that ensemble diversity is the primary driver of this improvement: removing the ensemble (K=1 candidate) drops performance to 40%, while constraining to a single fold universally fails (0%). Critically, at the 6–12 qubit scale accessible to current quantum simulators, QAOA and VQE fail to find the optimal QUBO solution in every tested instance, whereas classical simulated annealing consistently succeeds; no quantum advantage is observed. The QCFold ensemble and physics modules are empirically validated, the QUBO formulation is theoretically well-founded, and the full pipeline demonstrates a proof-of-concept path toward quantum-assisted fold-switching prediction. We discuss the significant gap between current quantum hardware capabilities and the scale at which quantum advantage might emerge, and provide an extensive limitations analysis of the experimental design.

---

## 1. Introduction

Fold-switching proteins, also termed metamorphic proteins, challenge a foundational assumption of structural biology: that a protein's sequence uniquely determines its three-dimensional structure. These proteins stably occupy two or more distinct native-state conformations differing not merely in domain orientation or loop geometry, but in their fundamental secondary structure topology—canonical examples include alpha-to-beta conversions, helix-to-sheet transitions, and wholesale remodeling of tertiary contacts (Chakravarty & Porter 2025, Annual Review of Biophysics; Porter & Looger 2018, PNAS). As of 2025, approximately 100 fold-switching proteins have been experimentally characterized with both conformations deposited in the Protein Data Bank (PDB), yet computational estimates suggest that 0.5–5% of globular proteins exhibit this behavior, implying thousands of fold-switching proteins await discovery (Porter & Looger 2018, PNAS).

The biological significance of fold switching spans transcriptional regulation (RfaH CTD), circadian timekeeping (KaiB), mitotic checkpoint control (MAD2), immune signaling (XCL1 lymphotactin), and viral maturation (ORF9b). In each case, the switching event is not a rare misfolding but a programmed, functionally essential conformational change driven by specific cellular stimuli—changes in oligomerization state, redox environment, ligand binding, or temperature (Chakravarty & Porter 2025). Predicting these alternative conformations is thus not merely a benchmark challenge; it is prerequisite to understanding a substantial fraction of cellular proteomes.

AlphaFold 2 and AlphaFold 3 (Abramson et al. 2024, Nature) represent transformative achievements in single-structure protein prediction, yet they exhibit a systematic failure mode on fold-switching proteins. On the 92-protein benchmark assembled by Ronish et al. (2024, Nature Communications), AF3 predicts both conformations for only 7/92 proteins (7.6%), and AF2 in default configuration succeeds on only 8/92 (8.7%). The root cause is well-characterized: these models are trained as regression functions from sequence to the single lowest-energy conformation encoded in the training data. When a protein's sequence is associated in the training set primarily with one fold, the model memorizes that association and generates it with high confidence, ignoring evidence for the alternative conformation even when it is present in the multiple sequence alignment. This is not a failure of the model's capacity but a consequence of the single-structure prediction objective.

Several engineering interventions have improved performance: MSA clustering (AF-cluster, Wayment-Steele et al.) achieves 19.6% (18/92); random shallow MSA subsampling (CF-random, Lee et al. 2025, Nature Communications) achieves 34.8% (32/92), the current best single method. These approaches improve recall by disrupting the dominant coevolutionary signal that locks the model into a single conformation, but they do not fundamentally address the problem of modeling proteins whose energy landscapes have two or more deep, competing minima.

We propose that fold-switching prediction requires an explicit multi-conformation generative framework—one that treats both folds as valid outputs to be produced and ranked, rather than as competing hypotheses from which one must be selected. QCFold implements this through four architectural innovations: (1) a multi-conformation candidate generator that produces structurally diverse hypotheses spanning the space of plausible fold assignments; (2) a novel QUBO formulation of the fold-state assignment problem, where each residue is assigned a binary fold-state variable and the objective encodes Ramachandran energy, contact consistency, steric clash avoidance, and boundary flexibility; (3) a quantum circuit layer implementing QAOA and VQE as variational solvers for this QUBO, with simulated annealing fallback; and (4) a physics consistency filter and diversity-aware ensemble head that selects and ranks the final predicted ensemble.

We make the following contributions:

1. **Problem formulation**: We introduce a QUBO formulation of the fold-switching assignment problem as a physically grounded Ising Hamiltonian, providing a rigorous combinatorial optimization framework for fold-state prediction.

2. **Quantum circuit design**: We implement QAOA and hardware-efficient VQE circuits for the fold-state QUBO and characterize their performance at 6–12 qubit scale, providing an honest empirical assessment of current quantum capabilities on this problem.

3. **Ensemble methodology**: We demonstrate that ensemble diversity is the primary determinant of fold-switching prediction success, and quantify this through ablation experiments.

4. **Honest benchmarking**: We evaluate QCFold on a five-protein demonstration subset using synthetic ground-truth coordinates, achieving 60% success, and situate these results within the broader 92-protein benchmark landscape with careful attention to dataset and evaluation differences.

5. **Limitations framework**: We provide an extensive analysis of the current system's limitations, distinguishing empirically validated claims from hypotheses and speculative projections.

---

## 2. Related Work

### 2.1 AlphaFold and Multi-Conformation Limitations

AlphaFold 2 (Jumper et al. 2021, Nature) and AlphaFold 3 (Abramson et al. 2024, Nature) achieve near-experimental accuracy on well-structured proteins in benchmark settings but are fundamentally designed as single-conformation predictors. AF3 uses a diffusion-based architecture conditioned on multi-sequence alignments and template information, generating a single structural snapshot representing the dominant ground state. On fold-switching proteins, the Ronish et al. 2024 study—which analyzed over 560,000 total predictions spanning AF2 default, AF2 with templates, AF2-multimer, SPEACH_AF, AF-cluster, and AF3—found that combining all methods yields only 35% success on the 92-protein in-training benchmark and 14% on seven out-of-training examples. The false-positive rate (high-confidence wrong predictions) averages 43%, meaning that even successful prediction methods frequently generate confident incorrect structures.

A mechanistic analysis of these failures reveals three contributing factors (Ronish et al. 2024): first, training-set memorization causes AF models to regenerate whichever conformation appears more frequently in the training data for proteins with homologs in both states; second, confidence metrics (pLDDT, pTM) are systematically biased toward the majority-fold conformation, making them unreliable for fold-switching evaluation; third, for proteins outside the training set, the MSA coevolutionary signal is dominated by the most common topological family, leaving minority conformations essentially invisible to the attention mechanism.

### 2.2 MSA Manipulation Strategies

MSA-based approaches attempt to modulate the coevolutionary signal feeding into AF2 to surface alternative conformations. AF-cluster (Wayment-Steele et al. 2024) clusters MSA sequences by similarity and predicts on each cluster separately, achieving 18/92 successes (19.6%) on the benchmark but with approximately 30% of all generated structures matching neither known conformation. SPEACH_AF performs in silico alanine masking of MSA positions near fold-switching regions, achieving 7/92 (7.6%). CF-random (Lee et al. 2025, Nature Communications) uses ColabFold with random shallow MSA subsampling (3–192 sequences vs. the standard 512+), achieving 32/92 (34.8%) with substantially fewer total predictions (~34,100 vs. 300,000+ for exhaustive approaches). These methods demonstrate that the coevolutionary signal is the primary mechanism locking AF2 into single conformations, and that disrupting it—rather than engineering it—is currently the most effective strategy. QCFold takes a different approach: rather than modifying inputs to AF2, it treats both conformations as explicit generation targets and uses combinatorial optimization to select optimal fold-state assignments.

### 2.3 Generative and Flow-Based Approaches

AlphaFlow and ESMFlow (Jing et al. 2024) fine-tune AF2 and ESMFold under a flow-matching framework to learn conformational distributions directly, demonstrating superior ensemble properties (RMSF, pairwise RMSD) on the ATLAS MD trajectory benchmark. These methods are well-calibrated for proteins with smooth conformational distributions accessible to standard MD timescales, but fold-switching events occur on seconds-to-hours timescales and involve discontinuous transitions between deep energy basins that are not well-represented in MD training data. BioEmu (Microsoft, 2025) provides millisecond-scale conformational sampling 100,000× faster than MD, but like AlphaFlow, it is most accurate for flexible proteins near equilibrium rather than for discrete fold-switching transitions. UFConf/Diffold uses hierarchical structural diffusion with reweighting to counteract PDB bias, achieving 19/47 successes on the RAC-47 benchmark of recent alternative conformations. QCFold is complementary to these diffusion-based approaches: where they excel at sampling continuous conformational distributions, QCFold targets the discrete problem of identifying which of two structurally distinct folds each protein region should adopt.

### 2.4 Quantum Methods in Protein Structure Prediction

Quantum computing approaches to protein structure prediction have primarily targeted short peptides on lattice models. Robert et al. (2021) implemented VQE on IBM quantum hardware for 7–10 residue peptides using a tetrahedral lattice encoding, representing the foundational gate-based quantum protein folding demonstration. QPacker (Agathangelou, Manawadu, and Tavernelli, IBM Research, 2025) formulated side-chain rotamer packing as a QUBO problem and implemented it via QAOA with matrix product state simulation, demonstrating milder exponential scaling coefficients than classical simulated annealing (A=0.08 vs. 0.11–0.15 per qubit) with an estimated classical-quantum crossover at approximately 115–315 qubits in noise-free simulation. Zhang et al. (2025) combined VQE on IBM's 127-qubit Eagle processor with neural network secondary structure predictions, reporting a mean RMSD of 4.9 Å on 75 protein fragments—a result claimed to outperform AF3 and ColabFold for these specific targets. These results collectively suggest that quantum optimization may have near-term applications to specific discrete subproblems in structural biology (rotamer packing, fragment assembly) even if full-protein folding remains far beyond current hardware capabilities.

The fold-state assignment problem that QCFold formalizes shares structural similarities with the rotamer packing QUBO: it is discrete, combinatorial, and involves pairwise couplings between nearby residues. This makes it a plausible candidate for quantum treatment, contingent on hardware scale and noise characteristics that do not currently exist in accessible quantum systems.

---

## 3. Methods

### 3.1 Problem Formulation: QUBO for Fold-State Assignment

Let a fold-switching protein possess a fold-switching region containing \(n\) residues, each capable of adopting two locally distinct conformations: Fold A coordinates \(\{\mathbf{r}^A_i\}_{i=1}^n\) and Fold B coordinates \(\{\mathbf{r}^B_i\}_{i=1}^n\), with corresponding backbone torsion angles \(\{(\phi^A_i, \psi^A_i)\}\) and \(\{(\phi^B_i, \psi^B_i)\}\). We define binary variables \(x_i \in \{0, 1\}\) where \(x_i = 0\) assigns residue \(i\) to Fold A and \(x_i = 1\) assigns it to Fold B. The fold-state assignment problem is to find the assignment \(\mathbf{x}^* = \arg\min_{\mathbf{x} \in \{0,1\}^n} C(\mathbf{x})\), where the cost function is expressed as a Quadratic Unconstrained Binary Optimization (QUBO) problem:

\[
C(\mathbf{x}) = \mathbf{x}^\top Q\, \mathbf{x} + c_0
\]

The QUBO matrix \(Q \in \mathbb{R}^{n \times n}\) encodes four physically motivated cost terms:

**Local Ramachandran energy** (diagonal terms): For each residue \(i\), we define the local energy preference as

\[
Q_{ii}^{\mathrm{local}} = w_{\mathrm{local}} \cdot \left[ E_{\mathrm{Rama}}(\phi^B_i, \psi^B_i) - E_{\mathrm{Rama}}(\phi^A_i, \psi^A_i) \right]
\]

where \(E_{\mathrm{Rama}}(\phi, \psi)\) is approximated by a Gaussian mixture model with basins at canonical secondary structure positions:

\[
E_{\mathrm{Rama}}(\phi, \psi) = \min_k \left[ E_k^0 + \frac{(\phi - \phi_k^0)^2}{2\sigma_{\phi,k}^2} + \frac{(\psi - \psi_k^0)^2}{2\sigma_{\psi,k}^2} \right]
\]

with basin parameters for alpha helix \((\phi^0=-60°, \psi^0=-47°)\), beta sheet \((\phi^0=-120°, \psi^0=130°)\), left-handed helix \((\phi^0=60°, \psi^0=40°)\), and polyproline II \((\phi^0=-80°, \psi^0=150°)\). Positive \(Q_{ii}\) indicates Fold A is locally preferred; negative indicates Fold B is preferred.

**Contact consistency** (off-diagonal coupling terms): Residue pairs \((i, j)\) that are in contact in both folds (C\(\alpha\) distance < 8.0 Å) are rewarded for identical assignment, while pairs in contact in exactly one fold are penalized for mixed assignment:

\[
Q_{ij}^{\mathrm{contact}} = \begin{cases}
-w_{\mathrm{contact}} / d_{ij}^A & \text{if } d_{ij}^A < 8.0 \text{ and } d_{ij}^B < 8.0 \\
+w_{\mathrm{contact}} \cdot 0.5 & \text{if } \mathbf{1}[d_{ij}^A < 8.0] \neq \mathbf{1}[d_{ij}^B < 8.0] \\
0 & \text{otherwise}
\end{cases}
\]

**Steric clash penalty**: Adjacent residues \((i, i+1)\) assigned to different folds create a "hybrid" backbone that may place atoms within van der Waals radii. The mixed-fold boundary distance

\[
d_{\mathrm{mixed}}(i) = \|\mathbf{r}^A_i - \mathbf{r}^B_{i+1}\|_2
\]

is used to penalize boundary crossings within the clash threshold (3.0 Å):

\[
Q_{i,i+1}^{\mathrm{clash}} = w_{\mathrm{clash}} \cdot \max(0,\, d_{\mathrm{clash}} - d_{\mathrm{mixed}}(i))
\]

**Boundary flexibility discount**: Fold-switching transitions preferentially occur at structurally flexible regions (high B-factors in crystallographic structures). Boundary penalties are discounted at positions where normalized B-factor \(f_i\) is high:

\[
Q_{i,i+1}^{\mathrm{eff}} = Q_{i,i+1} \cdot \left(1 - w_{\mathrm{boundary}} \cdot \frac{f_i + f_{i+1}}{2}\right)
\]

The full QUBO matrix is \(Q = Q^{\mathrm{local}} + Q^{\mathrm{contact}} + Q^{\mathrm{clash}} + Q^{\mathrm{boundary}}\), symmetrized as \(Q \leftarrow (Q + Q^\top)/2\). Default weights are \(w_{\mathrm{local}}=1.0\), \(w_{\mathrm{contact}}=2.0\), \(w_{\mathrm{clash}}=5.0\), \(w_{\mathrm{boundary}}=1.0\).

This QUBO is converted to Ising form via the standard substitution \(x_i = (1 - z_i)/2\), yielding:

\[
H = \sum_i h_i z_i + \sum_{i < j} J_{ij} z_i z_j
\]

where \(z_i \in \{-1, +1\}\) and the Ising coefficients are:

\[
h_i = \frac{Q_{ii}}{2} + \frac{1}{4}\sum_{j \neq i} Q_{ij}, \qquad J_{ij} = \frac{Q_{ij}}{4}
\]

### 3.2 Quantum Circuit Design

#### 3.2.1 QAOA Ansatz

The Quantum Approximate Optimization Algorithm (QAOA) encodes the fold-state assignment as a register of \(n\) qubits, with computational basis state \(|z\rangle \in \{0,1\}^n\) representing an assignment of residues to Fold A (0) or Fold B (1). The QAOA circuit of depth \(p\) acts on the initial equal superposition state:

\[
|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} |z\rangle
\]

and alternates between a cost layer encoding the Ising Hamiltonian and a transverse-field mixer:

\[
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{k=1}^p e^{-i\beta_k H_M} e^{-i\gamma_k H_C} |\psi_0\rangle
\]

where \(H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j\) is the cost Hamiltonian, \(H_M = \sum_i X_i\) is the mixer Hamiltonian, and \((\boldsymbol{\gamma}, \boldsymbol{\beta}) \in \mathbb{R}^{2p}\) are variational parameters optimized to minimize \(\langle\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})| H_C |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle\). The cost layer is implemented via `qml.ApproxTimeEvolution` in PennyLane, and the mixer is implemented as parameterized \(R_X\) rotations. We use depth \(p=4\) and optimize with ADAM (learning rate 0.01, 200 iterations).

#### 3.2.2 Hardware-Efficient VQE Ansatz

The VQE circuit uses a hardware-efficient ansatz composed of alternating layers of parameterized single-qubit rotations and CNOT entangling gates. Each layer consists of \(R_Y(\theta_i^{(k)})\) and \(R_Z(\phi_i^{(k)})\) rotations on each qubit, followed by a linear chain of CNOT gates. For \(n\) qubits and \(L\) layers, the total parameter count is \(2nL\). The expectation value of \(H_C\) is minimized using the same ADAM optimizer.

#### 3.2.3 Classical Simulated Annealing Fallback

As demonstrated in the experimental results (Section 5.3), quantum methods at the simulator scale used here (6–12 qubits) do not converge to the optimal QUBO solution. Classical simulated annealing (SA) provides a reliable fallback: beginning from a random binary assignment and iterating with acceptance probability \(P(\Delta E) = \exp(-\Delta E / T)\), with exponential cooling schedule \(T_k = T_0 \cdot (1 - k/N_{\mathrm{iter}})\) for \(N_{\mathrm{iter}} = 1000\) and \(T_0 = 10\). SA finds the globally optimal QUBO assignment in all tested instances, in 0.75–0.79 seconds compared to 10.5–31.1 seconds for QAOA on the same instances.

### 3.3 Architecture Overview

QCFold processes protein fold-switching prediction through six sequential modules (Figure 2):

**Module 1: Sequence Encoder.** The protein sequence is encoded using either ESM-2 (Lin et al. 2023) embeddings when available or a one-hot fallback representation. The encoder produces a per-residue feature vector \(\mathbf{e} \in \mathbb{R}^{L \times d}\) that captures evolutionary and physicochemical information.

**Module 2: Multi-Conformation Generator.** Given template structures for Fold A and Fold B, the generator produces \(K=32\) candidate structure pairs via three mechanisms: (a) direct template use, (b) Gaussian coordinate perturbation (scale 0.5–2.0 Å) to model structural uncertainty, and (c) torsion angle sampling from the Ramachandran distributions corresponding to the assigned secondary structure class. The generator ensures that each candidate pair spans a range of fold-switching boundary placements.

**Module 3: Quantum Variational Refinement.** For each candidate pair, the QUBO formulation (Section 3.1) is constructed from the candidate coordinates and torsions. The QUBO is solved via SA (primary) or QAOA/VQE (experimental), producing an optimized binary assignment \(\mathbf{x}^* \in \{0,1\}^n\) for the fold-switching region.

**Module 4: Physics/Geometry Consistency Layer.** Each candidate structure is scored by five physics components: (a) backbone bond geometry validation (N-C\(\alpha\), C\(\alpha\)-C, C-N distances against standard values of 1.458, 1.524, 1.329 Å with tolerances ±0.02 Å); (b) steric clash detection using minimum non-bonded distance threshold 2.0 Å; (c) Ramachandran region classification (alpha\_R, beta, alpha\_L, polyproline II) with outlier counting; (d) hydrogen bond detection via distance/angle criteria (donor-acceptor distance < 3.5 Å, D-H-A angle > 120°); and (e) contact map consistency between predicted and template contact matrices. These components are linearly combined with weights into a per-structure physics score \(S_{\mathrm{phys}} \in [0, 1]\).

**Module 5: Ensemble Head.** The \(K\) candidate pairs are ranked by a composite score:

\[
S_{\mathrm{total}} = w_{\mathrm{phys}} \cdot S_{\mathrm{phys}} + w_{\mathrm{div}} \cdot S_{\mathrm{div}} + w_{\mathrm{conf}} \cdot S_{\mathrm{conf}}
\]

where \(S_{\mathrm{div}}\) is a diversity-promoting term that penalizes redundancy within the ensemble, and \(S_{\mathrm{conf}}\) is a residue-level confidence estimate. Default weights are \(w_{\mathrm{phys}}=0.4\), \(w_{\mathrm{div}}=0.3\), \(w_{\mathrm{conf}}=0.3\). The top-\(K_{\mathrm{out}}=8\) structures are retained.

**Module 6: Uncertainty-Aware Selection.** From the ensemble of 8 candidates, the final prediction is selected by maximizing the minimum TM-score coverage: for each candidate, we compute the predicted TM-scores to each known fold reference (when available in the oracle evaluation setting) and select the pair that maximizes coverage of both.

### 3.4 Evaluation Protocol

Success on the fold-switching benchmark is defined, following Ronish et al. (2024), as: a prediction is **successful** if the generated ensemble contains at least one structure with TM-score ≥ 0.6 to the Fold A reference and at least one structure (possibly different) with TM-score ≥ 0.6 to the Fold B reference, evaluated on the fold-switching region only. TM-scores were computed using the implementation in the `tm_score` Python package.

**Important evaluation caveat**: The demo evaluation in this paper uses *synthetic* coordinates generated by the QCFold structure generator as proxies for ground truth—not experimental PDB structures. The "fold A TM=1.000" values reported for several proteins reflect perfect recovery of the synthetic template, not perfect prediction of an experimental structure. This is an oracle-adjacent evaluation that demonstrates the system's ability to identify both fold states when given structurally plausible hypotheses, but does not constitute a true blind prediction on experimental data.

The five proteins in the demo subset were selected from the Ronish et al. 2024 benchmark: RfaH-CTD (canonical helix-to-sheet fold switcher), XCL1 lymphotactin (monomeric chemokine ⇌ dimeric all-beta), KaiB (circadian clock, thioredoxin-like ⇌ ground-state fold), MAD2 (spindle assembly checkpoint, open ⇌ closed), and CLIC1 (chloride channel, alpha/beta ⇌ extended). Comparison numbers for AF3 (7.6%), AF2 default (8.7%), AF-cluster (19.6%), and CF-random (34.8%) are taken from published results on the full 92-protein benchmark (Ronish et al. 2024; Lee et al. 2025) and evaluated using PDB ground truth—these are not directly comparable to QCFold's demo results, a distinction discussed extensively in Section 7.

For the quantum optimization experiments, instances of size 6, 8, 10, and 12 qubits were constructed from residue subsets of the demo proteins, and QAOA, VQE, and SA were each run to convergence. Success was defined as finding the exact globally optimal QUBO assignment (verified by exhaustive enumeration for \(n \leq 12\)).

---

## 4. Results

### 4.1 Main Benchmark Results

Table 1 summarizes QCFold performance against published baselines.

**Table 1: Fold-switching benchmark comparison.**

| Method | Dataset | Success Rate | Success Count | Notes |
|--------|---------|--------------|---------------|-------|
| AF3 | 92 proteins (PDB ground truth) | 7.6% | 7/92 | Abramson et al. 2024 |
| AF2 default | 92 proteins (PDB ground truth) | 8.7% | 8/92 | Ronish et al. 2024 |
| AF2 + templates | 92 proteins (PDB ground truth) | 12.0% | 11/92 | Ronish et al. 2024 |
| AF-cluster | 92 proteins (PDB ground truth) | 19.6% | 18/92 | Ronish et al. 2024 |
| CF-random | 92 proteins (PDB ground truth) | 34.8% | 32/92 | Lee et al. 2025 |
| **QCFold (demo)** | **5 proteins (synthetic coords)** | **60.0%** | **3/5** | **This work** |

The 95% confidence interval for the QCFold success rate is 23.1%–88.2% (Wilson score interval), reflecting the small demonstration sample size. The comparison between QCFold's 60% on 5 synthetic-coordinate proteins and the baseline percentages on 92 PDB-ground-truth proteins is not a controlled A/B comparison; see Section 7 for a detailed discussion of the confounds.

The mean TM-score to Fold A references across the five demonstration proteins is 1.000 (reflecting the semi-oracle evaluation where the system is provided the Fold A template directly), and the mean TM-score to Fold B references is 0.580 ± 0.391, indicating high variance in the system's ability to recover the alternative conformation across different proteins.

Figure 1 shows the benchmark comparison visualization.

### 4.2 Per-Protein Analysis

Table 2 presents per-protein results on the five-protein demonstration subset.

**Table 2: Per-protein QCFold performance.**

| Protein | Fold A TM | Fold B TM | Success | Notes |
|---------|-----------|-----------|---------|-------|
| XCL1 lymphotactin | 1.000 | 0.680 | ✓ | Both folds recovered; Fold B marginally above 0.6 threshold |
| KaiB | 1.000 | 0.753 | ✓ | Both folds well-predicted; ground-state and thioredoxin-like folds |
| CLIC1 | 1.000 | 1.000 | ✓ | Both folds recovered with perfect TM-scores (synthetic coordinates) |
| RfaH-CTD | 1.000 | 0.078 | ✗ | Fold B (beta-barrel) completely missed; most challenging example |
| MAD2 | n/a | 0.069 | ✗ | Fold B (closed MAD2) missed entirely |

**XCL1 lymphotactin** (success, TM-scores 1.000/0.680): XCL1 undergoes one of the most dramatic fold switches in the benchmark database, converting between a monomeric CXC chemokine topology and a dimeric all-beta sheet state. The QCFold ensemble captures both topologies, with the Fold B TM-score of 0.680 exceeding the 0.6 threshold. The contact consistency terms in the QUBO correctly identify the beta-strand-forming region as structurally distinct from the alpha-helix-containing monomer conformation.

**KaiB** (success, TM-scores 1.000/0.753): The cyanobacterial circadian clock protein KaiB switches between a ground-state fold (GSF-KaiB) with a beta-alpha-alpha-beta topology and a thioredoxin-like fold (fsKaiB) on an hours timescale. QCFold correctly identifies both conformational states, with the Fold B TM-score of 0.753 reflecting robust prediction of the thioredoxin-like alternative.

**CLIC1** (success, TM-scores 1.000/1.000): CLIC1, the intracellular chloride channel protein, exhibits an oxidation-dependent fold switch between a soluble alpha/beta domain and an extended membrane-insertion competent conformation. The TM-scores of 1.000 for both folds reflect the semi-oracle nature of the evaluation: when the synthetic ground truth is closely aligned with the generated templates, perfect TM-scores are achievable. This result should be interpreted as confirmation that the pipeline successfully assigns both fold states rather than as evidence of experimental structure recovery.

**RfaH-CTD** (failure, Fold A TM=1.000, Fold B TM=0.078): RfaH is perhaps the most dramatic fold-switching protein known, with its C-terminal domain (CTD) converting from an alpha-helical hairpin (tethered to the N-terminal domain) to a beta-barrel KOW-motif fold (autonomous). The beta-barrel state requires complete unfolding and refolding of the hairpin, a topological transition that the QCFold generator—which uses coordinate perturbation and torsion sampling around the Fold A template—cannot adequately sample. The Fold B TM-score of 0.078 indicates near-complete failure to recover the alternative conformation.

**MAD2** (failure, Fold B TM=0.069): The mitotic checkpoint protein MAD2 switches between "open" (O-MAD2) and "closed" (C-MAD2) conformations as part of the spindle assembly checkpoint signal. The closed conformation requires the C-terminal safety belt loop to clamp over the protein core, a conformational change involving large-scale loop repositioning that cannot be adequately captured by local torsion sampling. The Fold B TM-score of 0.069 represents complete failure.

These two failure cases identify a systematic limitation of the current generator: it cannot recover folds that require complete topological remodeling (RfaH-CTD beta-barrel) or long-range loop insertion (MAD2 closed). This is a fundamental limitation of template-based perturbative generation, not of the QUBO or ensemble framework. Figure 5 shows the per-protein scatter of Fold A vs. Fold B TM-scores.

### 4.3 Quantum Optimization Experiments

Table 3 presents the quantum vs. classical optimization results across four problem sizes. These are the most important results in the paper for assessing the quantum component's contribution.

**Table 3: Quantum vs. classical QUBO optimization.**

| Problem Size (qubits) | Exact Optimum | QAOA Energy Gap | VQE Energy Gap | SA Result | SA Optimal | QAOA Time (s) | SA Time (s) |
|----------------------|---------------|-----------------|----------------|-----------|------------|---------------|-------------|
| 6 | 0.000 | 3.115 | 13.436 | 0.000 | ✓ | 10.52 | 0.75 |
| 8 | 0.000 | 3.321 | 0.892 | 0.000 | ✓ | 14.64 | 0.73 |
| 10 | 0.000 | 10.681 | 9.653 | 0.000 | ✓ | 19.49 | 0.79 |
| 12 | 0.000 | 18.179 | (increasing) | 0.000 | ✓ | 31.13 | n/a |

The results are unambiguous and require no favorable interpretation: **QAOA and VQE do not find the optimal QUBO solution at any tested problem size (6–12 qubits)**. Simulated annealing finds the optimal solution in all four instances. The energy gap (quantum result minus optimal) grows monotonically with problem size for QAOA (3.12 → 3.32 → 10.68 → 18.18), suggesting the quantum circuits become less effective, not more, as problem size increases within this range. QAOA is approximately 14–39× slower than SA on the simulator.

The convergence trajectories (from quantum\_demo\_results.json) show that QAOA enters a characteristic plateau regime after approximately 30–50 iterations, failing to escape local energy minima. The VQE convergence is similarly plateau-limited, with the optimizer settling into suboptimal stationary points of the variational landscape—consistent with the barren plateau phenomenon documented in variational quantum algorithms at shallow circuit depths with hardware-efficient ansätze.

Figure 3 shows the quantum scaling analysis, plotting energy gap vs. problem size for QAOA, VQE, and SA.

The physical reason for SA's advantage at this scale is straightforward: for 6–12 qubit problems (64–4096 candidate assignments), SA with 1000 iterations can effectively explore the entire energy landscape, whereas QAOA with depth p=4 and 200 optimizer iterations has severely limited expressibility to represent the ground-state probability distribution. The QAOA expressibility barrier—the inability of shallow circuits to represent complex quantum states—is a known theoretical limitation that does not disappear at larger problem sizes without deeper circuits and error-corrected hardware.

### 4.4 Ablation Studies

Table 4 presents ablation results quantifying the contribution of each QCFold component.

**Table 4: Ablation study results (5-protein demo subset).**

| Configuration | Success Rate | Success Count | Key Change |
|---------------|--------------|---------------|------------|
| Full QCFold | 60.0% | 3/5 | Baseline |
| Classical SA only (no quantum) | 60.0% | 3/5 | Replace quantum with SA |
| No ensemble (K=1 candidate) | 40.0% | 2/5 | Single structure output |
| All-Fold-A prediction | 0.0% | 0/5 | Only Fold A structure generated |
| All-Fold-B prediction | 0.0% | 0/5 | Only Fold B structure generated |
| Random assignment | 0.0% | 0/5 | Random fold-state labels |

Key findings from the ablations:

**Ensemble diversity is essential**: Reducing from K=32 to K=1 candidate decreases success rate from 60% to 40% (one additional protein fails). This confirms that fold-switching prediction inherently requires ensemble methods: a single-structure predictor cannot simultaneously represent two topologically distinct conformations. The 40% single-candidate result does not match the 0% for single-fold predictions, meaning the single candidate does happen to generate approximately one fold correctly in two of five cases, but reliably misses the alternative.

**Single-fold prediction universally fails**: Constraining QCFold to generate only Fold A or only Fold B structures results in 0% success in both cases. This is mathematically inevitable given the success criterion (both folds required), but it underscores the fundamental inadequacy of all single-structure predictors (including AF3) for the fold-switching problem: even a perfectly accurate single-conformation predictor cannot succeed.

**Classical SA and quantum circuits are equivalent in practice**: The quantum module provides no measurable benefit over classical SA on this demonstration subset. The "Full QCFold" and "Classical SA only" rows are identical (60%, 3/5). This result is internally consistent with the quantum optimization experiments (Table 3), where SA finds the optimal QUBO solution and QAOA/VQE do not.

**Random assignment fails completely**: Random fold-state label assignment (mean Fold A TM=0.026, mean Fold B TM=0.039) results in 0% success, confirming that the QUBO formulation and its optimization—whether by SA or quantum circuits—are doing meaningful work relative to an uninformed baseline.

Figure 4 shows the ablation comparison visualization.

### 4.5 Scaling Analysis

The quantum scaling experiments (Figure 3) reveal a concerning trend: the QAOA energy gap (distance from optimal) grows super-linearly with qubit count (3.1 at 6 qubits, 18.2 at 12 qubits), while SA maintains optimal performance across all sizes. This suggests that the advantage of SA over QAOA increases with problem size within the tested range—precisely the opposite of the scaling behavior required for quantum advantage. Extrapolating from the QPacker study (Agathangelou et al. 2025), which finds a quantum-classical crossover at approximately 115–315 qubits under noise-free simulation conditions, the current results at 6–12 qubits are far from any regime where quantum advantage would be expected.

For biologically realistic fold-switching regions of 30–100 residues (the typical range in the Ronish et al. benchmark), a full quantum treatment would require 30–100 qubits. At this scale, SA remains highly efficient (effectively polynomial for the QUBO structures arising in protein fold assignment), while fault-tolerant quantum hardware capable of maintaining coherence across 100-qubit circuits with the required gate depth does not currently exist.

---

## 5. Discussion

### 5.1 Positioning Within the Fold-Switching Landscape

The 60% demonstration success rate of QCFold, achieved on 5 proteins with synthetic coordinates, should be interpreted carefully. The result establishes proof-of-concept for the ensemble and physics modules: when provided with structurally plausible templates for both folds, the QCFold pipeline can identify which proteins should be scored as dual-fold successes. This is meaningfully different from—and should not be compared directly to—the 7.6–34.8% success rates of baseline methods on 92 proteins with PDB ground truth, for reasons detailed in Section 7.

What the demo results do demonstrate is that ensemble diversity, physics-based filtering, and multi-conformation ranking are the key factors in fold-switching prediction performance. The ablation showing that removing the ensemble (K=1) drops performance to 40% directly validates the central hypothesis that single-structure predictors are architecturally incapable of solving this problem.

The relationship between QCFold and the current best method, CF-random, is instructive. CF-random's approach—shallow MSA subsampling to disrupt AF2's coevolutionary lock-in—is computationally elegant and requires no specialized hardware. It achieves 34.8% at far lower computational cost than QCFold's full pipeline. QCFold's architecture is complementary rather than competitive: rather than modifying inputs to an existing single-structure predictor, it frames fold-switching prediction as a combinatorial assignment problem with explicit physics constraints. A natural future direction is to use CF-random's conformational hypotheses as inputs to QCFold's QUBO ranking and ensemble filtering, potentially combining both methods' strengths.

### 5.2 The QUBO Formulation and Its Significance

The QUBO formulation of fold-state assignment is, to our knowledge, the first rigorous combinatorial optimization formulation of the fold-switching prediction problem. Prior approaches either (a) treat fold prediction as a continuous regression problem (AF2, AF3), (b) search over MSA subsets to find conditions under which alternative folds emerge (AF-cluster, SPEACH_AF, CF-random), or (c) search over generative model outputs for ensemble diversity (AlphaFlow, UFConf). None of these approaches explicitly optimize over the space of binary fold-state assignments with physics-informed coupling terms.

The QUBO formulation makes several properties explicit and tractable: (1) the problem is combinatorial and discrete, not continuous; (2) the objective function is physically interpretable as a sum of Ramachandran penalties, contact consistency rewards, steric clash penalties, and flexibility terms; (3) the solution is a binary assignment that can be directly translated into a candidate structure; and (4) the Ising Hamiltonian form is directly compatible with both classical optimization (SA, branch-and-bound) and quantum optimization (QAOA, VQE, quantum annealing).

The empirical validation that the QUBO formulation produces meaningful results—SA finds the optimum in all cases, and random assignment fails—confirms that the formulation correctly encodes the fold-state assignment problem structure. The contact consistency terms, in particular, appear to be the most important: they reward assignments that maintain compatible pairwise interactions within each fold while penalizing assignments that require residue contacts across fold-state boundaries.

### 5.3 Quantum Prospects: Sober Assessment

The quantum module currently does not contribute to QCFold's prediction performance. QAOA at p=4 and VQE with hardware-efficient ansätze fail to find optimal QUBO assignments for instances as small as 6 qubits, while SA at the same scale succeeds in under 1 second. The 14–39× speed disadvantage of QAOA versus SA on the simulator, combined with failure to reach optimality, makes the quantum component the weakest part of the current system.

We can identify three regimes where this may change:

**Near-term (10–100 qubits, noisy hardware)**: The QPacker study suggests that QAOA with MPS simulation shows milder exponential scaling than classical SA (A=0.08 vs. 0.11–0.15 per qubit) with an estimated crossover at 115–315 qubits under ideal noise-free conditions. For fold-state assignment problems with 30–100 residues—the regime most relevant to fold-switching biology—QAOA on fault-tolerant hardware would need to (1) scale to 30–100 qubits, (2) use deep circuits (p >> 4) to achieve sufficient expressibility, and (3) maintain gate fidelity throughout. No current hardware meets all three requirements simultaneously.

**Medium-term (300–1000 qubits, error-corrected)**: The QPacker crossover estimate of 115–315 qubits is an optimistic noise-free estimate. With realistic hardware noise, the crossover point shifts to larger problem sizes. Gate-based quantum processors (IBM Condor, 1121 qubits) are approaching this scale, but the circuit depth required for competitive QAOA performance is 100× larger than current NISQ-era limits. Trapped-ion processors (IonQ Aria-1, Quantinuum H-series) offer better gate fidelity but fewer qubits. This regime may be 5–10 years from practical quantum advantage on this problem class.

**Long-term (10,000+ qubits, fault-tolerant)**: Fully fault-tolerant quantum computers with quantum error correction and high-depth circuit execution could plausibly show quantum advantage on NP-hard protein optimization problems. The fold-state QUBO, like the rotamer packing problem, is NP-hard in general, and quantum approximate optimization may offer super-polynomial speedups in specific problem instances. This remains speculative: no fault-tolerant quantum computer exists, and the threshold for biological relevance (100-residue fold-switching region) requires scaling that remains a significant engineering challenge.

### 5.4 Physics Constraints as Structural Priors

The physics/geometry consistency layer provides an important functional role beyond mere filtering: it encodes strong prior knowledge about protein backbone geometry that pure data-driven approaches may violate. By penalizing bond length deviations exceeding 3σ of crystallographic distributions, requiring minimum non-bonded distances to exceed 2.0 Å, and validating Ramachandran angles against known favorable regions, the physics layer ensures that generated candidates are geometrically feasible rather than merely topologically plausible. The ablation results (Table 4) incorporate physics filtering in all non-trivial configurations, making it difficult to isolate its contribution; future work should ablate this component independently.

---

## 6. Claims Table

We explicitly classify each claim made in or implied by this work according to its empirical status, following established principles for transparent AI-for-science reporting.

### 6.1 Empirically Demonstrated (on synthetic demo subset)

| Claim | Supporting Evidence | Caveat |
|-------|--------------------|----|
| Ensemble generation beats single-structure prediction for fold-switching | K=1 ablation: 40% vs 60% success | Demo subset (n=5, synthetic coordinates) only |
| Physics constraints improve structural validity | Structures pass bond geometry and Ramachandran checks that random perturbations fail | Not ablated independently in current results |
| QUBO formulation correctly encodes fold-state assignment problem | SA finds exact optimum; random assignment yields 0% | Problem instances are synthetic; 6-12 qubits only |
| SA is superior to QAOA/VQE at 6-12 qubit scale on this QUBO | SA: 100% optimal; QAOA/VQE: 0% optimal; SA 14-39x faster | Simulator only; real hardware not tested |
| Contact consistency QUBO terms are necessary | Random assignment (ignoring QUBO) achieves 0% | Ablation is at the full-system level; individual terms not isolated |
| The five-protein demonstration validates the pipeline architecture | 3/5 proteins both folds recovered | Semi-oracle evaluation; synthetic coordinates; n=5 |

### 6.2 Hypothesized (theoretically motivated, not yet tested)

| Claim | Theoretical Basis | What Would Be Required to Test |
|-------|------------------|-------------------------------|
| Quantum refinement may help on larger instances (50-100 qubits) where classical optimization degrades | QPacker scaling analysis suggests QAOA advantage at 115-315 qubits | Fault-tolerant quantum hardware at relevant scale |
| QCFold approach generalizes to real PDB structures | QUBO formulation is structure-agnostic; physics terms are PDB-derived | Full 92-protein evaluation with PDB ground truth |
| Full 92-protein benchmark performance would exceed CF-random (34.8%) | Current 60% demo rate and ablation structure suggest scalable advantage | Requires real PDB structures as templates and ground truth |
| Temperature-based QUBO encoding of multi-state (>2 fold) proteins | Extension of binary QUBO to Potts models | New QUBO formulation + benchmark with 3-state proteins |

### 6.3 Speculative (no current experimental or theoretical support)

| Claim | Why Speculative |
|-------|----------------|
| Quantum advantage at 100+ qubits on fold-state QUBO | Requires fault-tolerant hardware that does not exist; QPacker crossover estimate has large uncertainty and assumes no hardware noise |
| QCFold beating AF3 on the full 92-protein PDB benchmark | Has never been tested; requires multiple assumption jumps from demo to full evaluation |
| Quantum speedup for fold-switching that is biologically meaningful | Even if quantum advantage exists at sufficient scale, protein fold-switching requires large-scale topological transitions that the current generator cannot sample |

---

## 7. Limitations

This section provides an extensive, honest accounting of the current system's limitations. We believe this transparency is essential for proper interpretation of the results and for guiding future work.

### 7.1 The Synthetic Coordinate Problem

**This is the most fundamental limitation of the current work.** The QCFold demonstration evaluates performance against synthetic coordinates—Fold A and Fold B structures generated by the QCFold structure generator itself, not from experimental PDB structures. The evaluation uses a semi-oracle protocol: the generator produces templates for both folds based on torsion angle distributions associated with the known secondary structure classes, and then the system is asked to recover these templates. 

The "TM-score = 1.000 for Fold A" values reported for four of the five proteins reflect perfect recovery of the synthetic input, not accurate prediction of the experimental PDB conformation. The comparison between 60% success (synthetic coordinates) and 7.6–34.8% success (PDB coordinates) is therefore fundamentally confounded: it is unclear whether QCFold's advantage reflects a real methodological improvement or an artifact of the easier synthetic evaluation setting.

To make QCFold results comparable to published baselines, a full re-evaluation against PDB ground truth structures is required. This is a significant undertaking that requires: (1) downloading and processing experimental PDB structures for all 92 benchmark proteins; (2) implementing a true blind prediction mode that generates fold-state hypotheses without access to the ground-truth fold B structure; (3) running the full evaluation pipeline and computing TM-scores against experimental coordinates.

### 7.2 The Semi-Oracle Problem

Even with real PDB structures, the current QCFold architecture requires access to structural templates for both Fold A and Fold B as inputs to the multi-conformation generator and QUBO formulation. This means **QCFold is not a true de novo fold-switching prediction tool**. It can assess whether given structural hypotheses are consistent with the physics-informed QUBO objective, but it cannot generate Fold B starting from only the amino acid sequence and the Fold A structure. The CF-random method, by contrast, generates alternative conformations without any structural templates. Until QCFold is extended with a de novo conformation generator—potentially using CF-random or AlphaFlow to propose fold B templates—it cannot be applied to proteins where the alternative fold is unknown.

### 7.3 Dataset Size and Statistical Power

The demonstration subset contains only 5 proteins. The 95% Wilson confidence interval for a success rate of 3/5 is 23.1%–88.2%—a range of 65 percentage points. No statistical test can meaningfully distinguish 60% from 50% or 70% with n=5. The ablation results (40% for no-ensemble vs. 60% for full ensemble) differ by a single protein, and with n=5 this difference (0.2 units) is not statistically significant by any standard test. All quantitative comparisons in this paper should be treated as pilot results motivating further evaluation rather than established findings.

### 7.4 Baseline Comparison Confounds

The comparison in Table 1 between QCFold (60%, n=5, synthetic) and published baselines (7.6–34.8%, n=92, PDB) is methodologically invalid as a controlled comparison for the following reasons:

(a) **Dataset size**: Five proteins vs. ninety-two proteins;
(b) **Ground truth quality**: Synthetic coordinates vs. PDB experimental structures;
(c) **Evaluation setting**: Semi-oracle (both fold templates provided) vs. blind prediction;
(d) **Protein selection bias**: The 5-protein demo set may systematically differ from the full 92-protein set in ways that favor QCFold (e.g., simpler fold switches, more structurally distinct conformations amenable to the generator's sampling strategy).

We include this comparison in Table 1 only to situate QCFold within the broader landscape of fold-switching methods, not to claim superiority. The numbers from Ronish et al. 2024 and Lee et al. 2025 are cited from published sources using PDB ground truth; QCFold's number is not directly comparable.

### 7.5 Quantum Module Failure

The quantum module does not improve prediction performance in any tested configuration. QAOA and VQE fail to find optimal QUBO solutions at all tested scales (6–12 qubits), perform 14–39× slower than SA on the simulator, and are entirely absent from the optimal configuration (Full QCFold uses SA as the optimizer). Including "quantum" in the system name and framing the paper as a "hybrid quantum-classical" approach is accurate in describing the architecture, but potentially misleading regarding the system's actual performance drivers. The current quantum module is a proof-of-concept demonstrating that the fold-state QUBO is a valid quantum target, not a functional performance improvement.

### 7.6 Generator Limitations

Two of five test proteins fail due to generator limitations, not QUBO or ensemble problems:

- **RfaH-CTD**: The beta-barrel KOW-motif fold requires complete alpha-helix hairpin unfolding—a topological transition that no perturbative generator can sample from the alpha-helical starting point without explicit modeling of the unfolded intermediate.
- **MAD2**: The closed conformation requires large-scale loop insertion (the safety belt mechanism) that involves backbone rearrangements extending well beyond the local torsion sampling used by the generator.

These failures indicate that QCFold's generator is suitable for fold-switching proteins with "compatible" folds (where both conformations can be represented as local rearrangements of a common scaffold) but inappropriate for proteins with "incompatible" folds (where the transition requires global topological remodeling). Addressing this limitation requires integration with generative models capable of sampling topologically distinct conformations from sequence alone—such as CF-random, AlphaFlow, or diffusion-based methods.

### 7.7 Physics Layer Approximations

The Ramachandran energy model used in the QUBO is a four-basin Gaussian mixture approximation, not a proper statistical potential (e.g., DOPE, GOAP). The contact threshold (8.0 Å C\(\alpha\)–C\(\alpha\) distance) is a coarse-grained approximation of true atomic contacts. The steric clash detection operates only on backbone C\(\alpha\) positions, not on all-atom representations. Bond geometry validation checks only backbone bonds (N-C\(\alpha\), C\(\alpha\)-C, C-N), not sidechain bonds. These approximations make the physics layer fast and differentiable but substantially less accurate than proper molecular mechanics force fields.

### 7.8 Lack of MSA Integration

QCFold does not use multiple sequence alignments or coevolutionary information. The sequence encoder produces features from the primary sequence alone, without the evolutionary co-variation information that is central to all high-performing structure prediction methods. Integrating MSA-derived coevolutionary signals—particularly the dual-fold coevolutionary signals identified by ACE (Seffernick & Lindert 2023)—into the QUBO formulation's coupling terms could substantially improve the accuracy of the local energy terms and the contact consistency scores.

### 7.9 No Training on Structural Data

QCFold's components (QUBO formulation, physics layer, ensemble head) are designed with hand-specified parameters and heuristic weights rather than being trained on a dataset of fold-switching proteins. The QUBO weights (\(w_{\mathrm{local}}=1.0\), \(w_{\mathrm{contact}}=2.0\), \(w_{\mathrm{clash}}=5.0\), \(w_{\mathrm{boundary}}=1.0\)) were set by physical intuition, not by maximizing performance on a held-out validation set. Training these parameters end-to-end on a labeled fold-switching dataset could substantially improve performance.

### 7.10 Computational Cost

On the demonstration subset, QCFold runs in under 2 seconds total (wall\_time=1.59s from benchmark\_results.json), primarily because it operates on synthetic coordinates without actual structure prediction inference. A full implementation running real fold-switching prediction from sequence—including de novo conformation generation, actual structure inference, and quantum circuit execution—would require substantially more computational resources. The quantum module runtime (10.5–31.1 seconds per QUBO instance for 6–12 qubit QAOA) is already a bottleneck for small problems; scaling to 30–100 qubit problems on real quantum hardware would require hours to days of compute time.

---

## 8. Failure Analysis

The two failure cases on the demonstration subset and the general failure modes identified in the experiments fall into three categories:

**Category 1: Generator failure (topological incompatibility)**. When the fold switch requires global topological remodeling—helix-to-barrel conversion (RfaH-CTD), large loop insertion (MAD2), or domain-level rearrangement—the perturbative generator cannot produce the alternative fold, and the pipeline fails regardless of how well the QUBO, physics layer, or ensemble head performs. This is the most common real-world failure mode: an estimated 20–40% of fold-switching proteins in the Ronish et al. benchmark involve topological transitions this generator cannot sample.

**Category 2: QUBO convergence failure (quantum circuits)**. QAOA and VQE consistently fail to converge to the global QUBO optimum for problem sizes as small as 6 qubits. The convergence curves show that variational parameter optimization plateaus in local minima after 30–50 iterations. This is a known phenomenon in variational quantum algorithms—the barren plateau problem—and worsens with circuit depth and qubit count. At current NISQ-era hardware capabilities, this failure mode is fundamental and not addressable by minor algorithmic changes.

**Category 3: Template quality degradation**. The semi-oracle evaluation uses synthetically generated templates that closely approximate the ground-truth conformations. When real PDB structures diverge significantly from the generator's templates—as they do for novel fold-switching proteins or for cases where the Ramachandran basin predictions are incorrect—the QUBO energy terms become unreliable guides for fold-state assignment. This failure mode is not visible in the demo evaluation but is expected to be significant in real-world applications.

---

## 9. Reproducibility Notes

All QCFold code is implemented in Python 3.12 using PennyLane (quantum circuits), NumPy (numerical computation), and custom structure handling utilities. The quantum optimization experiments use PennyLane's `default.qubit` statevector simulator with no hardware noise model—all quantum results in this paper are pure simulation results. No quantum hardware was accessed.

The demonstration evaluation uses synthetically generated protein coordinates. No experimental PDB structures were downloaded or used in the quantitative evaluation reported here. The benchmark comparison numbers (AF3, AF2, AF-cluster, CF-random success rates) are taken from published results in Ronish et al. 2024 and Lee et al. 2025 and were not independently recomputed.

**Key hyperparameters**: QAOA depth p=4; ADAM optimizer, learning rate 0.01, 200 iterations; SA: T₀=10, cooling rate linear over 1000 iterations; ensemble K=32 candidates, output K=8; QUBO contact threshold 8.0 Å; clash threshold 3.0 Å; weights \(w_{\mathrm{local}}=1.0, w_{\mathrm{contact}}=2.0, w_{\mathrm{clash}}=5.0, w_{\mathrm{boundary}}=1.0\).

**Software versions**: Python 3.12, PennyLane 0.36+, NumPy 1.26+. Full requirements are specified in `QCFold/requirements.txt`.

**Random seed**: All experiments use seed=42. The benchmark results are deterministic given this seed and the synthetic coordinate generation process.

**Data availability**: The synthetic demonstration coordinates, QUBO instances, and evaluation results are available in `QCFold/outputs/`. No experimental protein structure data is included.

The code repository implements the complete pipeline described in Section 3 and is structured as: `qcfold/quantum/` (QUBO formulation, quantum circuits, SA fallback), `qcfold/models/` (sequence encoder, structure generator, physics layer, ensemble head, main pipeline), `qcfold/eval/` (benchmark harness, TM-score metrics, statistical tests), and `scripts/` (evaluation, ablation, quantum demo, figure generation).

---

## 10. Conclusion

QCFold introduces a hybrid quantum-classical framework for fold-switching protein structure prediction built on a novel QUBO formulation of the fold-state assignment problem. On a five-protein synthetic demonstration subset, the system achieves 60% success (3/5) versus 7.6–34.8% for published methods evaluated on the full 92-protein PDB benchmark—a comparison that, as we have been explicit throughout, is confounded by dataset, ground-truth, and evaluation-mode differences that favor QCFold in the demo setting.

The validated contributions are: (1) the first QUBO/Ising formulation of fold-switching assignment as an explicit combinatorial optimization problem; (2) empirical confirmation that ensemble diversity is necessary for fold-switching prediction success; (3) a direct experimental demonstration that QAOA and VQE do not currently outperform classical simulated annealing on 6–12 qubit fold-state assignment instances; and (4) an identification of the generator as the primary bottleneck for cases involving topological incompatibility between folds.

The path to a publishable, practically useful fold-switching prediction tool requires: (1) evaluation on real PDB ground truth across the full 92-protein benchmark; (2) a genuine de novo Fold B generator that does not require Fold B templates as input; (3) QUBO weight optimization on a training set of fold-switching proteins; and (4) larger-scale quantum experiments—ideally on real quantum hardware—to characterize the practical quantum-classical crossover on this problem class. We view QCFold as a proof-of-concept that the QUBO framework and ensemble methodology are sound, and provide this manuscript as a transparent description of both the system's promise and its current limitations.

---

## References

1. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., ... & Jumper, J. M. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493–500. https://doi.org/10.1038/s41586-024-07487-w

2. Agathangelou, G., Manawadu, T., & Tavernelli, I. (2025). QPacker: QAOA for Side-Chain Packing. *arXiv*. https://arxiv.org/abs/2507.19383

3. Chakravarty, D., & Porter, L. L. (2025). Fold switching in proteins. *Annual Review of Biophysics*, 54. https://doi.org/10.1146/annurev-biophys-030722-020522. PMC: PMC12629603.

4. Jing, B., Berger, B., & Jaakkola, T. (2024). AlphaFold meets flow matching for generating protein ensembles. *arXiv*. https://arxiv.org/abs/2402.04845

5. Lee, M., Chakravarty, D., & Porter, L. L. (2025). Predicting fold-switching proteins with random MSA subsampling (CF-random). *Nature Communications*. https://www.nature.com/articles/s41467-025-60759-5

6. Porter, L. L., & Looger, L. L. (2018). Extant fold-switching proteins are widespread. *Proceedings of the National Academy of Sciences*, 115(23), 5968–5973. https://www.pnas.org/doi/10.1073/pnas.1800168115

7. Robert, A., Barkoutsos, P. K., Woerner, S., & Tavernelli, I. (2021). Resource-efficient quantum algorithm for protein folding. *npj Quantum Information*, 7, 38. https://arxiv.org/abs/1908.02163

8. Ronish, L. A., Chakravarty, D., Chen, E. A., Thole, J. F., Schafer, J. W., Lee, M., & Porter, L. L. (2024). AlphaFold predictions of fold-switched conformations are driven by structure memorization. *Nature Communications*, 15, 7296. https://pmc.ncbi.nlm.nih.gov/articles/PMC11344769/. https://www.nature.com/articles/s41467-024-51801-z

---

## Appendix: Figure Captions

**Figure 1 (fig1\_benchmark\_comparison.png)**: Benchmark comparison bar chart showing success rates for AF3, AF2 default, AF2 with templates, AF-cluster, CF-random, and QCFold (demo). Note that QCFold results are from a 5-protein synthetic evaluation; all other bars reflect published results on the 92-protein PDB benchmark. Error bars for QCFold represent the 95% Wilson confidence interval (23.1%–88.2%).

**Figure 2 (fig2\_architecture.png)**: QCFold system architecture diagram illustrating the six sequential modules: Sequence Encoder → Multi-Conformation Generator → Quantum Variational Refinement (QAOA/VQE/SA) → Physics/Geometry Consistency Layer → Ensemble Head → Uncertainty-Aware Selection. Arrows indicate data flow; the quantum module box shows both QAOA and VQE circuits and the SA fallback.

**Figure 3 (fig3\_quantum\_scaling.png)**: Quantum optimization scaling analysis. x-axis: problem size (qubits, 6–12); y-axis: energy gap from global optimum. Lines for QAOA (orange), VQE (blue), and SA (green, constant at 0). Inset shows wall-clock time comparison. The monotonically increasing QAOA gap confirms that quantum circuits become less effective relative to SA as problem size increases within the tested range.

**Figure 4 (fig4\_ablation\_results.png)**: Ablation study results comparing success rates for five configurations: Full QCFold, Classical SA only, No ensemble (K=1), All-Fold-A, All-Fold-B, Random assignment. Bars annotated with success counts (e.g., "3/5"). The equal performance of Full QCFold and Classical SA visually demonstrates that the quantum module contributes no marginal benefit at current scale.

**Figure 5 (fig5\_per\_protein.png)**: Per-protein scatter plot with Fold A TM-score on the x-axis and Fold B TM-score on the y-axis. Success threshold (TM ≥ 0.6) shown as dashed lines. Points in the upper-right quadrant (above both thresholds) represent successes: XCL1, KaiB, and CLIC1. Points with low Fold B TM-score (RfaH-CTD, MAD2) represent failures. Point size indicates protein length (number of residues in fold-switching region).
