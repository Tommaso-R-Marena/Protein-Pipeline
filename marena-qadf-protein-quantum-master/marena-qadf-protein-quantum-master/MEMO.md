# Research Memo

**TO:** Research Advisor / Review Committee  
**FROM:** Tommaso Marena, Departments of Chemistry and Philosophy, The Catholic University of America  
**RE:** QADF Protein Side-Chain Optimization — Novelty, Risk, and Evidence Assessment  
**DATE:** 2025  

---

## 1. What This Project Is

The Quantum Amenability Decision Framework (QADF) is a structured scoring rubric for assessing whether a given computational subproblem in structural biology is a viable near-term target for quantum optimization. The central observation motivating the framework is that "quantum-enhanced protein structure prediction" is often discussed in terms of the full folding problem, which remains far beyond near-term quantum hardware. A more productive question is: which *subproblems* within structural biology have properties — small instance size, sparse local connectivity, discrete variable structure, classical hardness at relevant scales — that make them genuinely suitable for quantum treatment in the 50–200 qubit regime that current and near-future devices inhabit? The QADF provides a nine-dimensional rubric for scoring subproblems along these axes and produces a ranked ordering that can guide research prioritization.

As a proof-of-concept, this project applies the QADF to **side-chain rotamer optimization**: given a fixed protein backbone, assign low-energy rotamer states (discrete χ angle bins from the Dunbrack library) to each side chain such that pairwise steric and electrostatic clash is minimized. This subproblem has a natural QUBO encoding — one-hot binary variables over rotamer states, quadratic coupling terms from pairwise interaction energies — and the 4-residue window instances studied here (12 binary variables, ~12 qubits) sit at the upper edge of what is currently simulable classically with reasonable compute time. Both classical baselines (exhaustive search, greedy assignment, simulated annealing) and classically simulated quantum experiments (QAOA at circuit depths p=1 and p=2 via PennyLane `default.qubit`) are benchmarked on PDB structures 1L2Y and 1UBQ. The project further develops a hybrid ML model with an equivariant graph neural network backbone and a calibrated per-residue confidence head, producing scores analogous to AlphaFold's pLDDT.

The deliverable of this project is threefold: (1) the QADF rubric itself, documented and validated; (2) a working prototype demonstrating the full pipeline from PDB structure to QUBO to quantum circuit to calibrated confidence output; and (3) an honest scaling analysis documenting where classical simulation breaks down and what that implies for the prospects of near-term quantum advantage on this problem class. The prototype does not claim quantum advantage and is not expected to — its purpose is to demonstrate that the pipeline is sound, the QADF rubric is discriminating, and the evidence base for the framework is grounded in current literature.

---

## 2. Why This Is Non-Trivial for an Independent Undergraduate

Each component of this project requires graduate-level background to implement correctly. The QUBO formulation demands fluency in combinatorial optimization, Lagrangian penalty construction, and the specific conventions of the Dunbrack rotamer library. The QAOA implementation requires understanding of parameterized quantum circuits, variational hybrid algorithms, and the subtleties of cost-function landscape geometry that determine whether COBYLA or gradient-based optimizers converge. The equivariant GNN architecture requires knowledge of SE(3)-equivariant message-passing and the graph representation of protein contact networks. The calibrated confidence estimation requires familiarity with reliability diagrams, Expected Calibration Error, and the distinction between accuracy and calibration. This project connects all four in a single coherent pipeline — an integration task that is independently non-trivial from any one domain's perspective. The connection itself — asking whether quantum optimization outputs can carry calibrated uncertainty estimates analogous to those in learned structure prediction — appears not to have been previously demonstrated in the literature in this form.

This work was completed independently, without external funding, laboratory access, or QPU hardware. These constraints are not apologized for; they are documented and reflected in the scope of the claims.

---

## 3. Novelty Assessment

The literature most directly relevant to this work is as follows:

**Doga et al. (2024)** [DOI: 10.1021/acs.jctc.4c00067] presents a framework for quantum-classical hybrid approaches to biomolecular simulation and discusses the conditions under which quantum optimization may offer advantage. It does not provide a multi-dimensional scoring rubric for quantum readiness (the QADF), does not include calibrated confidence scores on optimization outputs, and does not demonstrate the rotamer optimization QUBO pipeline.

**Agathangelou et al. (2025)** [arXiv:2507.19383] is the closest prior work: it presents a QUBO formulation for side-chain rotamer optimization and benchmarks QAOA on small instances. It does not include an ML hybrid model, does not address confidence estimation or calibration, and does not provide a generalizable scoring framework for comparing multiple subproblems (the QADF). This work is properly cited throughout and the overlap is explicitly acknowledged.

**Bauza et al. (2023)** [DOI: 10.1038/s41534-023-00733-5] demonstrates QAOA on peptide conformation optimization, establishing that quantum approaches to molecular conformation problems are technically feasible at small scales, but does not extend to side-chain packing specifically or to uncertainty quantification.

**What this work adds:** The combination of (1) the QADF rubric as a reusable, multi-dimensional quantum readiness scoring tool, (2) the hybrid ML model with calibrated confidence output, (3) the honest reporting of QAOA performance against all classical baselines on real PDB data, and (4) the explicit documentation of classical simulation scaling limits collectively constitute a contribution that is distinct from any single prior work. The framework is the primary contribution; the experiments validate it.

---

## 4. Evidence Strength Assessment

| Claim | Strength | Basis |
|---|---|---|
| Rotamer optimization is a better near-term quantum target than global backbone folding | **Moderate** | Supported by QADF rubric scores + published literature on qubit requirements; not independently experimentally confirmed |
| QAOA p=1/p=2 solution quality vs. exhaustive ground truth on 12-qubit instances | **Strong** | Directly computed in this work's classically simulated experiments |
| Classical simulation becomes intractable beyond ~20–25 qubits under repeated experiment conditions | **Strong** | Directly observed; consistent with published PennyLane benchmarks |
| Depolarizing noise at ε=0.01 degrades QAOA solution quality by the reported percentage | **Moderate** | Computed in this work but using injected synthetic noise, not hardware noise characterization |
| QADF correctly classifies 8 subproblems on 9 dimensions | **Preliminary** | Rubric scoring is the author's own judgment applied to literature; no independent validation |
| Per-residue confidence calibration improves ECE relative to uncalibrated scores | **Moderate** | Directly measured in this work; limited by small training set and proxy-based ground truth |
| AlphaFold 2 achieves median CASP14 GDT_TS of 92.4 | **Strong** | Published data from Jumper et al. 2021 [DOI: 10.1002/prot.26257], Table 1B |
| AlphaFold 2 does not report side-chain χ1/χ2 recovery in the CASP14 paper | **Strong** | Verified by direct inspection of Jumper et al. 2021; the paper reports GDT_TS and LDDT, not rotamer recovery rates |

---

## 5. Risks

**QAOA underperformance is expected and is a negative result, not a failure.** At the 12-qubit, 4-residue scale studied here, exhaustive search and simulated annealing both find the ground state trivially. QAOA at p=1 and p=2 does not outperform greedy assignment on any window studied. This result is entirely consistent with the known limitations of shallow QAOA on small, low-depth instances and is reported transparently throughout. The framing is that this negative result *supports* the QADF's diagnostic value: the rubric correctly predicts that near-term quantum advantage for rotamer optimization requires larger, denser instances than are currently simulable classically.

**Classical simulation imposes a hard ceiling.** Statevector simulation via PennyLane `default.qubit` becomes computationally prohibitive beyond approximately 20–25 qubits on the hardware available for this project. All quantum claims are therefore restricted to instances within this range. No extrapolation to hardware performance is made.

**Experimental scope is narrow.** All quantum experiments are restricted to the 4-residue, 12-qubit window. No quantum experiments are performed on the full 1L2Y or 1UBQ structures. The broader QADF conclusions rest on the QUBO analysis and published literature, not on large-scale quantum experiments.

**The prototype does not compete with FlowPacker.** FlowPacker [bioRxiv: 2024.07.05.602280] is a state-of-the-art deep learning approach to side-chain packing that substantially outperforms all methods tested in this project, including simulated annealing. This is stated explicitly. The prototype's value is as a quantum methods testbed and QADF validation exercise, not as a competitive structure prediction tool.

**No experimental validation beyond computational benchmarks.** No laboratory validation of predicted rotamer assignments is performed or claimed. All assessments of solution quality are relative to computed force-field energies, not experimental structures beyond what is directly encoded in the PDB reference coordinates.

---

## 6. Why This Is Still Worth Publishing or Presenting

The scientific value of this project lies not in demonstrating quantum advantage — which would require hardware and scale unavailable here — but in three areas where an independent contribution can be made with existing resources.

First, the QADF rubric is a genuine conceptual contribution. The quantum computing literature lacks a systematic, multi-dimensional scoring tool for assessing the quantum readiness of computational biology subproblems. Providing one, validating it against known results, and demonstrating its discriminating power across eight subproblems is useful to the community regardless of the prototype's scale.

Second, the honest reporting of QAOA's limitations at small scales is itself valuable. Much of the popular discourse around quantum-enhanced protein structure prediction understates how far near-term devices are from tractable instances. A careful, reproducible negative result from a clean classical simulation baseline is a contribution.

Third, the connection between quantum optimization outputs and calibrated confidence estimation has not been demonstrated in this form in the literature. Even if the calibration is preliminary, establishing the pipeline and the methodology is worth documenting.

For a project completed independently at the undergraduate level, the appropriate target is not a flagship journal publication claiming breakthrough results. The appropriate target is a venue where the QADF framework's intellectual content is assessed on its own terms, the honest scope is respected, and the reproducibility of the work is valued.

---

## 7. Recommended Next Steps

1. **Extend the quantum experiments to larger windows.** A 6-residue, 18-qubit instance would allow a more definitive comparison between QAOA and simulated annealing and would stress-test the QUBO encoding conventions. This remains within classical simulation range if restricted to a small number of shots, and would strengthen the empirical basis for the QADF scoring.

2. **Obtain independent review of the QADF rubric scoring.** The nine-dimension scoring of eight subproblems is currently the author's own assessment applied to the literature. Having one or more domain experts score the same subproblems independently would allow inter-rater reliability to be assessed and would strengthen the rubric's claim to objectivity.

3. **Replace the B-factor proxy with a higher-quality ground-truth disorder signal.** The current confidence calibration uses crystallographic B-factors as a proxy for residue-level disorder. Replacing this with order parameters from molecular dynamics ensembles (e.g., from pre-computed RCSB conformational ensemble data) would provide a more rigorous calibration target and is achievable without wet-lab resources.

---

*This memo is provided for internal review purposes. All claims above are documented in CLAIMS.md with supporting evidence ratings.*
