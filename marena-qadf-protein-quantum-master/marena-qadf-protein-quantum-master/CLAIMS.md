# Claims Registry

**Project:** Quantum Amenability Decision Framework (QADF) for Protein Side-Chain Rotamer Optimization  
**Author:** Tommaso Marena, The Catholic University of America  
**Purpose:** This document registers every substantive scientific claim made in the project, with explicit support level, evidence type, and key caveat. It is intended to be read alongside the paper and to facilitate honest external review.

**Support Levels:**
- **Strongly Supported** — multiple independent lines of evidence; claim would survive removal of any single piece
- **Moderately Supported** — consistent with available evidence but has meaningful uncertainty or limitations
- **Preliminary** — based on limited or indirect evidence; reasonable hypothesis, not yet robustly established
- **Speculative** — extrapolation beyond direct evidence; stated as a conjecture, not a finding

**Evidence Types:**
- **Published literature** — based on peer-reviewed or preprint sources (see citation)
- **This work's experiments** — based on experiments run in this project (note: all quantum experiments are classically simulated)
- **This work's simulations [CLASSICALLY SIMULATED]** — quantum circuit results produced by PennyLane `default.qubit`; no QPU hardware used
- **Theoretical argument** — logical/mathematical derivation without direct empirical confirmation

---

## Claims Table

| # | Claim | Support Level | Evidence Type | Key Caveat |
|---|---|---|---|---|
| 1 | Side-chain rotamer optimization is a better near-term quantum target than global backbone folding | **Moderately Supported** | Published literature + this work's QADF rubric scoring | The QADF rubric scoring reflects the author's structured assessment of the literature, not an independently validated quantitative metric; the ranking could shift with different weighting of rubric dimensions. Published sources (Doga 2024 [DOI: 10.1021/acs.jctc.4c00067]; Khatami 2023 [DOI: 10.1371/journal.pcbi.1011033]) support the general conclusion that discrete, local subproblems are more tractable near-term targets, but do not directly compare these two subproblems. |
| 2 | QAOA on a 4-residue, 12-qubit QUBO instance achieves a final energy within the observed gap above the exhaustive-search ground truth (values reported per-window in results/benchmarks/) | **Strongly Supported** | This work's simulations [CLASSICALLY SIMULATED] | Results are for ideal noiseless simulation only. QAOA p=1 and p=2 do not outperform greedy assignment on any window studied; this is a negative result, reported honestly. Instance size (12 qubits) is far below the scale at which quantum advantage is expected or claimed. Energy values are in units of the QUBO objective function, not physical kcal/mol without the parametric conversion described in the supplementary methods. |
| 3 | Classical quantum circuit simulation (statevector, PennyLane `default.qubit`) becomes computationally intractable beyond approximately 20–25 qubits under repeated experimental conditions (multiple QAOA runs, multiple windows) | **Strongly Supported** | This work's simulations [CLASSICALLY SIMULATED] | "Intractable" here means impractical for repeated experiments on the hardware used (consumer laptop / standard cloud compute); it does not mean that a single large-system simulation is impossible with HPC resources. The ceiling is a practical constraint of this project, not a theoretical limit. |
| 4 | Depolarizing noise at ε = 0.01 (applied after each QAOA gate layer) degrades QAOA solution quality by the percentage reported in results/benchmarks/noise_experiment.csv | **Moderately Supported** | This work's simulations [CLASSICALLY SIMULATED] | Noise is injected synthetically via PennyLane's `depolarizing` channel; it does not capture the correlated, non-Markovian noise structure of real QPU hardware. The reported degradation percentage is specific to the instances and circuit depths studied (p=1, 12 qubits). The actual degradation on hardware would likely be larger and less predictable. |
| 5 | The QADF framework correctly classifies 8 structural biology subproblems using 9 scoring dimensions, producing a ranked ordering consistent with the current quantum computing literature | **Preliminary** | This work's QADF rubric (published literature + author judgment) | "Correctly classifies" means the ranking is consistent with the author's reading of the literature and with theoretical arguments about quantum tractability. No independent reviewer has scored the same subproblems using the same rubric. Inter-rater reliability has not been assessed. The rubric is a structured tool, not a validated psychometric instrument. |
| 6 | This project's per-residue confidence scores show lower Expected Calibration Error (ECE) after Platt scaling than before calibration, when evaluated against B-factor-derived disorder proxies | **Moderately Supported** | This work's experiments | Calibration is measured against crystallographic B-factors as a proxy for residue disorder; B-factors conflate thermal motion, crystal contacts, and data quality, making them an imperfect ground truth. The GNN model is trained on limited data. The calibration improvement is real but its biological interpretability is uncertain. |
| 7 | AlphaFold 2 achieves a median CASP14 GDT_TS score of 92.4, with 58 of 92 CASP14 domains achieving GDT_TS > 90; the T1047s1-D1 domain is the noted failure case (GDT_TS = 50.47) | **Strongly Supported** | Published literature — Jumper et al. 2021, Table 1B [DOI: 10.1002/prot.26257]. **Based on published data, not re-run.** | These numbers are taken directly from Jumper et al. 2021, Table 1B. They have not been independently recomputed. They refer specifically to the CASP14 assessment conditions; AlphaFold 2 performance on other benchmark sets or under different conditions may differ. |
| 8 | AlphaFold 2 does not report side-chain χ1 or χ2 rotamer recovery rates in the primary CASP14 assessment paper (Jumper et al. 2021) | **Strongly Supported** | Published literature — Jumper et al. 2021 [DOI: 10.1002/prot.26257]. **Based on published data (verified by direct inspection), not re-run.** | This claim is verified by direct reading of the Jumper et al. 2021 paper, which reports GDT_TS, lDDT, and related backbone metrics. The absence of reported χ1/χ2 recovery in that specific paper does not mean AlphaFold 2 lacks side-chain prediction capability; subsequent publications and the AlphaFold Multimer paper include side-chain analyses. This claim is restricted to the single cited paper. |
| 9 | Low pLDDT regions in AlphaFold 2 predictions are strongly associated with intrinsically disordered regions (IDRs) and flexible loops, as has been shown by post-hoc structural analysis | **Moderately Supported** | Published literature (Jumper et al. 2021 [DOI: 10.1002/prot.26257]; Khatami et al. 2023 [DOI: 10.1371/journal.pcbi.1011033]). **Based on published data, not re-run.** | The association is well-documented in the literature and is the design intent of pLDDT as a confidence metric. However, the causal direction (low pLDDT *causes* the association vs. IDRs *produce* low pLDDT because the model lacks training signal) matters for interpretive claims and is not always distinguished. This claim is stated in the observational, correlational sense. |
| 10 | FlowPacker outperforms previous side-chain packing baselines (including SCWRL4 and Rosetta-based methods) across most standard metrics, as reported in the FlowPacker preprint | **Moderately Supported** | Published literature — FlowPacker preprint [bioRxiv: 2024.07.05.602280]. **Based on published data, not re-run.** | This claim is taken from the FlowPacker authors' own benchmark tables. Head-to-head comparisons in machine learning papers are subject to benchmark selection effects and may not generalize to all protein families or data splits. FlowPacker's performance substantially exceeds the prototype developed in this project, and this gap is acknowledged explicitly throughout the paper. |

---

## Additional Claims (Secondary)

| # | Claim | Support Level | Evidence Type | Key Caveat |
|---|---|---|---|---|
| 11 | One-hot encoding of rotamer states with 3 bins per residue produces 12 binary variables for a 4-residue window, requiring a 12-qubit quantum register for direct QUBO embedding | **Strongly Supported** | Theoretical argument (combinatorics) | Assumes exactly 3 rotamer bins per residue; some residue types have more or fewer meaningful rotamer states in the Dunbrack library. The 3-bin simplification is a deliberate design choice, not a claim about the full rotamer space. |
| 12 | The one-hot penalty term with λ = 5.0 effectively enforces valid rotamer assignments (exactly one bin per residue active) in the QUBO solutions found by classical optimizers | **Moderately Supported** | This work's experiments | Effectiveness is assessed empirically on the instances studied; the choice of λ = 5.0 is heuristic and may not generalize to larger windows or different energy scales. |
| 13 | Simulated annealing (10 random restarts, scipy implementation) finds the exhaustive-search ground state for all 4-residue windows studied in 1L2Y and 1UBQ | **Moderately Supported** | This work's experiments | This is an empirical observation for the specific instances and temperature schedules used. It is not a guarantee; harder instances or different schedules could produce different results. |
| 14 | The Dunbrack backbone-dependent rotamer library (PubMed: 21645855) provides the most widely used discrete rotamer state definitions for computational side-chain packing | **Moderately Supported** | Published literature — Dunbrack 2011 [PubMed: 21645855] | "Most widely used" is a qualitative assessment based on citation frequency and adoption in standard tools (Rosetta, SCWRL). It is not independently quantified here. |

---

## Versioning Note

This claims registry was prepared at the time of initial submission. Any updates to experimental results, revised numerical values, or changes to the scope of claims following peer review should be reflected in an updated version of this file with a dated changelog.
