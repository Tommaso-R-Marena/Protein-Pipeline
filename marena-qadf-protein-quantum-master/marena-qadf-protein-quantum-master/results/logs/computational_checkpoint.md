# Computational Checkpoint
## QADF Project: Hybrid Quantum-Classical Protein Structure Prediction
## Target: Side-Chain Rotamer Optimization
## Generated after completion of Phases 1–10

**CRITICAL NOTE: ALL QUANTUM RESULTS ARE [CLASSICALLY SIMULATED]**
Platform: PennyLane default.qubit v0.44.1 (classical statevector simulation)
No QPU hardware used in any experiment.

---

## Complete File Inventory

### /paper/ — Scientific Documents

| File | Phase | Content | Size |
|---|---|---|---|
| `phase1_positioning_memo.md` | 1 | Problem definition, novelty gap, AlphaFold gap analysis, dataset selection | ~10KB |
| `phase2_qadf.md` | 2 | QADF scoring table (8 subproblems × 9 criteria), decision tree | ~16KB |
| `phase3_model_spec.md` | 3 | EGNN + PQC architecture, loss function, training config, ablation plan | ~16KB |
| `phase8_alphafold_comparison.md` | 8 | AF2 CASP14 numbers, comparison table, fairness caveat | ~8KB |
| `phase10_dynamic_or_limits.md` | 10 | Energy landscape analysis, limitations, resource boundaries | ~10KB |

### /data/pdb/ — PDB Structure Files

| File | Description |
|---|---|
| `1L2Y.pdb` | TC5b Trp-cage mini-protein, 20 residues (NMR). Downloaded from RCSB PDB. Used for quantum experiments. |
| `1UBQ.pdb` | Ubiquitin, 76 residues. Downloaded from RCSB PDB. Used for classical baseline. |

### /data/rotamers/ — Processed Rotamer Data

| File | Description |
|---|---|
| `1L2Y_rotamers.csv` | Per-residue features: seq, phi/psi, chi1/chi2, rotamer bins, CA coords (20 rows) |
| `1UBQ_rotamers.csv` | Per-residue features: seq, phi/psi, chi1/chi2, rotamer bins, CA coords (76 rows) |
| `splits.json` | Train/val/test split specification (70/15/15, sequence identity clustering) |

**1L2Y rotamer stats:**
- 20 residues total; 17 with chi1; 10 with chi2
- Chi1 bin distribution: g−=7, t=4, g+=6
- Missing chi1: 3 (Gly×3)

### /data/qubo/ — QUBO Encoding Files

| File | Description |
|---|---|
| `qubo_matrix.npy` | 12×12 QUBO matrix Q for 4-residue window (TYR3-ILE4-GLN5-TRP6) |
| `encoding_metadata.json` | Encoding decisions, residue details, QUBO statistics |
| `all_energies.json` | Energies for all 81 valid configurations (exhaustive enumeration) |

**Key QUBO result:**
- Ground truth assignment [t, g−, t, t] achieves global minimum energy −34.0651
- Ground truth rank: 1/81 (optimal configuration)

### /data/setup_and_data.py — Phase 4 Script
Installs and verifies packages; downloads and processes 1L2Y, 1UBQ from RCSB PDB API.

### /data/qubo_encoding.py — Phase 5 Script
Builds 12×12 QUBO matrix; verifies ground truth is global minimum; saves matrix and metadata.

### /results/experiments.py — Phase 6 Script
Classical baselines (exhaustive, greedy, SA) and quantum experiments [CLASSICALLY SIMULATED] (QAOA p=1,2, scaling study, noise analysis).

### /results/confidence_analysis.py — Phase 7 Script
Per-residue confidence estimation, calibration metrics, bootstrap CIs, Wilcoxon test.

### /results/visualization.py — Phase 9 Script
Generates 10 publication figures as PNG (300dpi) and PDF.

### /results/energy_landscape.py — Phase 10 Script
1D chi1 torsion angle energy scan (5-degree resolution, −180° to +180°).

### /results/benchmarks/ — Benchmark Result Files

| File | Content |
|---|---|
| `classical_results.json` | Exhaustive, greedy, SA results on 4-residue QUBO instance |
| `qaoa_results.json` | QAOA p=1,2 results [CLASSICALLY SIMULATED] including convergence history |
| `scaling_study.json` | QAOA simulation time vs. window size (n=2..6 residues) [CLASSICALLY SIMULATED] |
| `noise_analysis.json` | QAOA under depolarizing noise (p=0, 0.001, 0.01) [CLASSICALLY SIMULATED] |
| `per_residue_confidence.csv` | Per-residue confidence scores, predicted states, chi1 errors (20 rows) |
| `calibration_metrics.json` | ECE, reliability diagram, Pearson/Spearman correlations |
| `bootstrap_cis.json` | Bootstrap 95% CIs for chi1 MAE, rotamer accuracy, ECE (n=1000 resamples) |
| `phase_summary.json` | Complete phase-by-phase summary of all outputs and key results |

### /results/figures/ — Publication Figures (20 PNG + 20 PDF = 21 files)

| File | Description |
|---|---|
| `fig1_pipeline.png/pdf` | Hybrid quantum-classical pipeline diagram |
| `fig2_qadf_taxonomy.png/pdf` | QADF 8-subproblem scoring table |
| `fig3_scaling_study.png/pdf` | QAOA simulation time vs. window size + qubit count |
| `fig4_noise_degradation.png/pdf` | Energy degradation under depolarizing noise [CLASSICALLY SIMULATED] |
| `fig5_reliability_diagram.png/pdf` | Calibration reliability diagram (ECE=0.015) |
| `fig6_confidence_vs_error.png/pdf` | Confidence score vs. chi1 error scatter (Pearson r=−0.31) |
| `fig7_confidence_profile.png/pdf` | Per-residue confidence profile for 1L2Y (pLDDT color scheme) |
| `fig8_qaoa_convergence.png/pdf` | QAOA p=1,2 convergence histories [CLASSICALLY SIMULATED] |
| `fig9_benchmark_comparison.png/pdf` | Solution quality and runtime comparison bar chart |
| `fig10_confidence_distribution.png/pdf` | Confidence distribution comparison (this project vs. AF2 schematic) |
| `energy_landscape.png/pdf` | 1D chi1 torsion energy landscape for TYR/ILE/GLN/TRP |
| `captions.md` | Full figure captions for all 11 figures |

### /results/logs/ — Log Files

| File | Content |
|---|---|
| `phase0_literature_verified.md` | 15 verified references with DOIs/arXiv IDs |
| `environment.txt` | Package version list |
| `computational_checkpoint.md` | This file — complete file inventory and key results |

---

## Key Numerical Results Summary

### Phase 5: QUBO Encoding
- 4-residue window (TYR3, ILE4, GLN5, TRP6) from 1L2Y
- Ground truth [t, g−, t, t] = **global minimum** at E = −34.0651 (rank 1/81)
- QUBO matrix: 12×12, 78 non-zero elements, penalty P=10

### Phase 6: Optimization Experiments [CLASSICALLY SIMULATED]
| Method | Energy | Matches GT | Runtime |
|---|---|---|---|
| Exhaustive search | −34.0651 | ✓ | 0.0003s |
| Greedy | −34.0651 | ✓ | 0.00006s |
| Simulated Annealing | −34.0651 | ✓ | 0.0056s |
| QAOA p=1 [CS] | 93.8285 | ✗ | 23.3s |
| QAOA p=2 [CS] | 179.3853 | ✗ | 33.0s |

**Scaling boundary**: 18 qubits (n=6 residues) → 125s per optimization

**Noise analysis** [CLASSICALLY SIMULATED]:
- ε₂ = 0 (noiseless): 0% degradation
- ε₂ = 10⁻³: 2.6% degradation
- ε₂ = 10⁻²: 23.3% degradation

### Phase 7: Confidence & Calibration
| Metric | Value | 95% CI |
|---|---|---|
| Chi1 MAE | 19.5° | [12.9°, 26.2°] |
| Rotamer accuracy | 100% | [100%, 100%] |
| Mean confidence score | 98.5 | [97.6, 99.3] |
| ECE | 0.0148 | [0.0072, 0.0242] |
| Wilcoxon (classical < QAOA) | p = 0.034 | Significant at α=0.05 |
| Pearson r (conf vs. error) | −0.310 | (p = 0.226) |

---

## Literature References Used

All references verified in Phase 0. See /results/logs/phase0_literature_verified.md for full details.

| ID | Citation |
|---|---|
| REF-01 | Doga et al. 2024, JCTC, DOI: 10.1021/acs.jctc.4c00067 |
| REF-02 | Agathangelou et al. 2025, arXiv: 2507.19383 |
| REF-03 | Khatami et al. 2023, PLOS CB, DOI: 10.1371/journal.pcbi.1011033 |
| REF-04 | Jumper et al. 2021, Proteins, DOI: 10.1002/prot.26257 |
| REF-05 | Dunbrack 2011, Proteins, DOI: 10.1002/prot.22921 |
| REF-06 | Bauza et al. 2023, npj QI, DOI: 10.1038/s41534-023-00733-5 |
| REF-07 | FlowPacker 2024, bioRxiv, DOI: 10.1101/2024.07.05.602280 |
| REF-08 | CalPro 2026, arXiv: 2601.07201 |
| REF-11 | NISQ noise parameters (emergentmind.com) |
| REF-12 | Hybrid VQC ML 2025, arXiv: 2502.11951 |
| REF-13 | AlphaFold pLDDT color convention |
| REF-14 | AlphaFold-Metainference, DOI: 10.1038/s41467-025-56572-9 |
| REF-15 | ENGINE/EGNN, Genome Biology 2025, PMC12665208 |

---

## Important Caveats

1. **All quantum results are [CLASSICALLY SIMULATED]** — no QPU hardware used
2. **QAOA at p=1,2 does not find optimal solution** — consistent with REF-06 findings
3. **Model not end-to-end trained** — Phase 7 uses energy-based inference, not learned weights
4. **Dataset is small** (20 residues for quantum; 76 for classical) — results are proof-of-concept
5. **100% rotamer accuracy on 1L2Y** reflects both the well-ordered structure and simple 3-bin encoding
6. **AlphaFold comparison** is not a direct numerical comparison — different tasks, different benchmarks

---

*Checkpoint completed after all phases (1–10) executed successfully.*
*All output files verified to exist in /home/user/workspace/marena-qadf/*
