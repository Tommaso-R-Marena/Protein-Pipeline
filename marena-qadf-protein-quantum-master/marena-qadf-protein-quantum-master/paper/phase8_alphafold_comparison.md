# Phase 8 — AlphaFold 2 Comparison Table
## Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)

---

## Mandatory Fairness Caveat

> **Fairness note**: AlphaFold 2 numbers are taken from Jumper et al. (2021), DOI: 10.1002/prot.26257 (Table 1A/B), and apply to the CASP14 benchmark under conditions not directly comparable to this study. This project addresses only side-chain rotamer optimization on PDB structures of ≤25 residues using classical simulation of quantum circuits. It does not predict global backbone folds. Direct numerical comparison of GDT_TS values is not appropriate. AlphaFold was not re-run for this study; all comparison numbers are sourced from the published paper.

---

## 1. Published AlphaFold 2 CASP14 Performance Numbers

All numbers below are sourced directly from **Jumper et al. 2021** (Proteins 2021;89(12):1711–1721, DOI: 10.1002/prot.26257, PMC: PMC9299164), Tables 1A and 1B.

| Metric | Value | Source |
|---|---|---|
| Median domain GDT_TS (best-of-5 models) | **92.4** | Table 1B |
| Domains with GDT_TS > 90 (best-of-5) | **58 / 92** (63%) | Table 1B |
| Mean domain GDT_TS (all 5 models) | **87.32** | Table 1A |
| Mean domain GDT_TS (top-1 by pLDDT) | **88.01** | Table 1B |
| pLDDT model ranking improvement | **86%** of max possible | Table 1B |
| T1044 TM-score (full sequence) | **0.960** (vs. 0.807 original) | Table 1B |
| Failure case: T1047s1-D1 GDT_TS | **50.47** (β-sheet at wrong angle) | Table 1B |
| Side-chain χ1/χ2 recovery rates | **NOT REPORTED** in this paper | See note below |

**Note on χ1/χ2 side-chain recovery**: The primary AlphaFold 2 CASP14 evaluation paper (DOI: 10.1002/prot.26257) does not report side-chain χ1 or χ1+2 recovery rates. The AF2 Nature paper (DOI: 10.1038/s41586-021-03819-2) similarly focuses on backbone accuracy metrics (GDT_TS, TM-score, lDDT-Cα). Side-chain accuracy for AF2 has been reported in follow-up benchmarking studies (e.g., CASP14 assessors' analysis), but those numbers are not used in this comparison to avoid attributing results to a different study.

---

## 2. Comparison: AlphaFold 2 vs. This Project

The table below compares the two approaches on dimensions where legitimate comparison is possible, and explicitly marks dimensions where comparison is **not applicable (N/A)**.

| Dimension | AlphaFold 2 (CASP14) | This Project (QADF Rotamer) | Comparable? |
|---|---|---|---|
| **Task** | Global backbone fold prediction | Side-chain rotamer optimization (fixed backbone) | ❌ Different tasks |
| **Backbone GDT_TS (median)** | 92.4 | N/A — backbone not predicted | ❌ N/A |
| **Backbone GDT_TS (failure case T1047s1-D1)** | 50.47 | N/A | ❌ N/A |
| **Side-chain χ1 recovery** | Not reported | 100% bin accuracy on 1L2Y (17 residues) | ⚠️ Partial — different benchmark |
| **Mean χ1 MAE** | Not reported | 19.5° ± 3.5° (95% CI: [12.9°, 26.2°]) | ⚠️ Cannot compare — no AF2 number |
| **Per-residue confidence score** | pLDDT (backbone metric; miscalibrated per REF-08) | Confidence score calibrated to χ1 accuracy | ✓ Conceptually comparable, different metric |
| **Calibration (ECE)** | pLDDT not calibrated; 15–25% coverage degradation [REF-08] | ECE = 0.0148 ± 0.0045 (95% CI: [0.007, 0.024]) | ✓ Both report confidence; ECE directly comparable |
| **Dataset** | CASP14 (92 domains, full-length proteins) | 1L2Y (20 residues), 1UBQ (76 residues) | ❌ Different scale |
| **Quantum component** | None (classical DL) | QAOA [CLASSICALLY SIMULATED] on 12-qubit Ising | ❌ Not comparable |
| **Disordered regions** | Low pLDDT; no ensemble [REF-14] | Not addressed in this project | ❌ N/A |
| **Hardware** | ~128 TPUs (v3-512 pod) | CPU / classical simulation | ❌ Not comparable |

---

## 3. AlphaFold Gap Analysis

### 3a. The Backbone-Only Confidence Problem

AlphaFold 2's pLDDT score regresses **per-residue lDDT-Cα** — a metric based on Cα atom distances, not side-chain atoms. It does not provide:
- Per-residue χ1 or χ2 accuracy estimates
- Confidence intervals for side-chain dihedral angles
- Calibrated probability that a given side-chain placement is within 40° of the true χ1

CalPro (arXiv: 2601.07201, REF-08) demonstrates that pLDDT exhibits **15–25% coverage degradation** under distribution shift — i.e., the nominal confidence intervals for structural predictions do not hold when proteins are outside the training distribution. This motivates the explicit calibration regularizer in our model.

### 3b. Disordered Region Limitations

AlphaFold 2 assigns low pLDDT (<50) to intrinsically disordered regions (IDRs), which represent a genuine biological challenge. AlphaFold-Metainference (Nature Communications 2025, DOI: 10.1038/s41467-025-56572-9, REF-14) explicitly addresses this by using AF2 backbone distances as restraints in molecular dynamics simulations — acknowledging that AF2 alone cannot model IDR ensembles. Side-chain placement in IDRs is therefore doubly uncertain: neither backbone nor side-chain positions are reliably predicted.

### 3c. The Side-Chain Performance Gap

The most relevant comparison — side-chain χ1/χ2 accuracy — cannot be made numerically from the primary AlphaFold 2 papers because these metrics are not reported. State-of-the-art classical side-chain packers (e.g., FlowPacker, REF-07, bioRxiv DOI: 10.1101/2024.07.05.602280) represent the relevant classical baseline for side-chain accuracy, not AlphaFold 2's backbone-focused architecture.

**What this project specifically addresses that AlphaFold 2 does not**:
1. Discrete rotamer optimization via quantum-inspired combinatorial search
2. Calibrated per-residue confidence specifically for χ1 angle accuracy (not backbone lDDT-Cα)
3. Explicit [CLASSICALLY SIMULATED] quantum optimization of the discrete rotamer assignment problem
4. Bootstrap confidence intervals and statistical tests for all reported metrics

---

## 4. Contextual AlphaFold 2 Performance Data

The following information is provided for scientific context, not for direct numerical comparison with this project.

### CASP14 Summary (REF-04, DOI: 10.1002/prot.26257)

| Condition | Mean GDT_TS |
|---|---|
| All 5 models (target average) | 87.32 |
| Top-1 model selected by pLDDT | 88.01 |
| Best-of-5 (oracle selection) | ~92+ |
| Median domain (best-of-5) | 92.4 |

The improvement from 87.32 to 88.01 by using pLDDT for model selection demonstrates that pLDDT carries signal for backbone quality — it accounts for **86% of the maximum possible improvement** from oracle selection. This is the strongest evidence in favor of pLDDT's utility, but it is a *relative* ranking tool, not a calibrated probability.

### T1047s1-D1 Failure Case

Domain T1047s1-D1 (GDT_TS = 50.47) is described in REF-04 as a large β-sheet domain where the final system predicted a β-strand packed at the wrong angle. This exemplifies a category of prediction failures where even state-of-the-art backbone prediction methods make large errors, reinforcing that backbone prediction is not a solved problem — and that improving downstream tasks (such as side-chain placement) conditional on backbone coordinates is a useful research direction.

---

## 5. Summary

| Comparison Point | Finding |
|---|---|
| AF2 backbone accuracy | Best-in-class (median GDT_TS 92.4 on CASP14) |
| AF2 side-chain accuracy | Not reported in primary papers |
| pLDDT calibration | Miscalibrated under distribution shift (15–25% coverage degradation, REF-08) |
| This project's task | Orthogonal: side-chain rotamers, not backbone |
| This project's confidence calibration | ECE = 0.015 ± 0.005 on 1L2Y test set |
| Direct numerical comparison | Not appropriate — different benchmarks, tasks, and scales |

The primary scientific contribution of this project is not to outperform AlphaFold 2, but to demonstrate a proof-of-concept quantum-hybrid approach to a complementary subproblem (side-chain rotamer optimization) with calibrated uncertainty — a capability not present in the published AlphaFold 2 architecture.

---

*References: REF-04 (Jumper et al. 2021, DOI: 10.1002/prot.26257), REF-07 (FlowPacker, DOI: 10.1101/2024.07.05.602280), REF-08 (CalPro, arXiv: 2601.07201), REF-14 (AlphaFold-Metainference, DOI: 10.1038/s41467-025-56572-9)*
