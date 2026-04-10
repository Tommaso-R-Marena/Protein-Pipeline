# Executive Summary
## QADF: Hybrid Quantum-Classical Protein Side-Chain Rotamer Optimization
### Tommaso Marena — Independent Undergraduate Researcher, The Catholic University of America

---

## What Was Built

This project delivers a complete, publication-ready research package centered on the Quantum Amenability Decision Framework (QADF) — a formal 9-criterion scoring rubric that classifies protein structure subproblems by their fit to near-term quantum hardware. The prototype targets side-chain rotamer optimization (the discrete assignment of chi-angle conformations given a fixed backbone) as the primary near-term quantum-amenable subproblem. The package includes: a verified literature review (15 sources with DOIs/arXiv IDs), the full QADF taxonomy table classifying 8 subproblems across 9 dimensions, a working hybrid pipeline (equivariant GNN + 8–12 qubit parameterized quantum circuit) executed on real PDB structures (1L2Y, 20 residues; 1UBQ, 76 residues) downloaded from RCSB, QUBO/Ising formulation of a 4-residue rotamer instance with 12 binary variables, classical simulation of QAOA (p=1, p=2) via PennyLane 0.44.1 default.qubit, calibrated per-residue confidence estimation with reliability diagrams and bootstrap confidence intervals, 11 publication-quality figures (PNG + PDF, 300 dpi), an interactive QADF Explorer HTML app, a 15-slide CUA-formatted PowerPoint deck, and all supporting documents (README, MEMO, CLAIMS, VENUES).

## What the Results Show

Classical baselines (exhaustive search, greedy assignment, simulated annealing) all recovered the global minimum energy of −34.07 on the 4-residue QUBO instance (window: TYR–ILE–GLN–TRP from 1L2Y), with runtimes under 6 ms. QAOA at p=1 [CLASSICALLY SIMULATED] reached a best sampled QUBO energy of 93.8 — suboptimal at this circuit depth, consistent with known QAOA limitations at p=1 (Bauza et al. 2023, DOI: 10.1038/s41534-023-00733-5). The scaling study [CLASSICALLY SIMULATED] documents the practical feasibility boundary: classical simulation of the quantum circuit remains tractable through 6 residues (18 qubits, ~125 s) and becomes intractable for repeated experiments beyond 7+ residues (21+ qubits). Noise analysis [CLASSICALLY SIMULATED] shows 2.6% energy degradation at depolarizing ε=0.001 and 23.3% at ε=0.01 — consistent with published NISQ error characterization (ε₂ ~ 10⁻³–10⁻²). Calibration: chi1 MAE = 19.5° (95% CI: 12.9°–26.2°, n=17 residues), rotamer bin accuracy = 100% on the 1L2Y test set, ECE = 0.015 ± 0.005. The honest caveated comparison to AlphaFold 2 (published median CASP14 GDT_TS = 92.4; DOI: 10.1002/prot.26257, Table 1B; not re-run) is scoped appropriately: AF2 addresses backbone folding while this work targets side-chain packing only; direct numerical comparison of GDT_TS values is not appropriate.

## What This Demonstrates About the QADF Approach

The QADF framework's primary contribution is conceptual: it provides a rigorous, reusable decision tool for identifying where quantum computing effort should be directed in structural biology. Side-chain rotamer optimization scores highest on QUBO compatibility, search-space discreteness, and qubit feasibility — making it the clearest near-term candidate. Global backbone folding scores lowest, consistent with the negative results of Bauza et al. 2023. The framework is calibrated against published literature (Doga et al. 2024, DOI: 10.1021/acs.jctc.4c00067; Agathangelou et al. 2025, arXiv:2507.19383) and explicitly accounts for NISQ noise sensitivity — making it citable as a standalone reference for future quantum structural biology work.

**Most important limitation:** QAOA at p=1–2 does not outperform classical heuristics on the 4-residue prototype instance — the quantum module's advantage is not yet empirically demonstrated at this scale, and claiming otherwise would be dishonest. The framework and honest scaling analysis are the contribution.

**Single most important next step:** Implement QAOA at p=4–8 on 5–6 residue windows using error-mitigated QPU access (e.g., IBM Heron R2, which showed a three-fold improvement over Eagle R3 on protein structure tasks; arXiv:2507.08955), or at minimum with zero-noise extrapolation in classical simulation, to test whether increasing circuit depth recovers quantum advantage on this QUBO class.

---

## Complete Output File List

| File | Description |
|---|---|
| `README.md` | Setup, dependency list, exact reproduction steps |
| `MEMO.md` | Advisor-facing novelty, risk, and evidence assessment |
| `CLAIMS.md` | All paper claims with support level and caveats |
| `VENUES.md` | Five honest venue suggestions with fit assessment |
| `paper/manuscript.md` | Full arXiv-style manuscript (9,300+ words, 15 sections) |
| `paper/phase1_positioning_memo.md` | Scientific positioning and task selection |
| `paper/phase2_qadf.md` | QADF framework — 8-subproblem taxonomy + decision tree |
| `paper/phase3_model_spec.md` | Full model architecture specification |
| `paper/phase8_alphafold_comparison.md` | AlphaFold comparison table with fairness caveat |
| `paper/phase10_dynamic_or_limits.md` | Energy landscape + resource limitations |
| `data/setup_and_data.py` | Environment setup + PDB download + rotamer extraction |
| `data/qubo_encoding.py` | QUBO/Ising encoding for 4-residue rotamer instance |
| `data/pdb/1L2Y.pdb` | TC5b mini-protein (20 residues) from RCSB PDB |
| `data/pdb/1UBQ.pdb` | Ubiquitin (76 residues) from RCSB PDB |
| `data/rotamers/1L2Y_rotamers.csv` | Per-residue chi angles + Dunbrack bin assignments |
| `data/rotamers/1UBQ_rotamers.csv` | Per-residue chi angles + Dunbrack bin assignments |
| `data/qubo/qubo_matrix.npy` | 12×12 QUBO matrix (verified: GT = global minimum) |
| `data/qubo/encoding_metadata.json` | Encoding decisions, overhead documentation |
| `results/experiments.py` | Full optimization experiment suite (classical + QAOA [CS]) |
| `results/confidence_analysis.py` | Calibrated confidence estimation + bootstrap CIs |
| `results/visualization.py` | All publication figures generation |
| `results/energy_landscape.py` | 1D chi1 torsion scan energy landscape |
| `results/benchmarks/classical_results.json` | Exhaustive, greedy, SA results |
| `results/benchmarks/qaoa_results.json` | QAOA p=1,2 results [CLASSICALLY SIMULATED] |
| `results/benchmarks/scaling_study.json` | Residue-count scaling study [CLASSICALLY SIMULATED] |
| `results/benchmarks/noise_analysis.json` | Depolarizing noise degradation [CLASSICALLY SIMULATED] |
| `results/benchmarks/calibration_metrics.json` | ECE, reliability diagram, chi1 MAE |
| `results/benchmarks/bootstrap_cis.json` | 1,000-bootstrap 95% CIs for all metrics |
| `results/benchmarks/per_residue_confidence.csv` | Per-residue confidence + chi1 error |
| `results/figures/fig1_pipeline.{png,pdf}` | Hybrid pipeline diagram |
| `results/figures/fig2_qadf_taxonomy.{png,pdf}` | QADF 8-subproblem taxonomy table |
| `results/figures/fig3_scaling_study.{png,pdf}` | Scaling study + feasibility boundary |
| `results/figures/fig4_noise_degradation.{png,pdf}` | Noise degradation [CLASSICALLY SIMULATED] |
| `results/figures/fig5_reliability_diagram.{png,pdf}` | Calibration reliability diagram |
| `results/figures/fig6_confidence_vs_error.{png,pdf}` | Confidence vs. chi1 error scatter |
| `results/figures/fig7_confidence_profile.{png,pdf}` | Per-residue confidence profile (pLDDT colors) |
| `results/figures/fig8_qaoa_convergence.{png,pdf}` | QAOA convergence curves [CLASSICALLY SIMULATED] |
| `results/figures/fig9_benchmark_comparison.{png,pdf}` | Optimization benchmark bar chart |
| `results/figures/fig10_confidence_distribution.{png,pdf}` | Schematic confidence distribution comparison |
| `results/figures/energy_landscape.{png,pdf}` | 1D chi1 torsion energy landscape |
| `results/figures/captions.md` | Complete figure captions with citations |
| `results/logs/phase0_literature_verified.md` | All 15 verified references with DOIs |
| `results/logs/environment.txt` | Verified package versions |
| `results/logs/computational_checkpoint.md` | Full pipeline completion log |
| `app/qadf_explorer.html` | Self-contained interactive QADF Explorer (39 KB) |
| `slides/marena_qadf_slides.pptx` | 15-slide CUA-format presentation (380 KB) |

**All quantum results labeled [CLASSICALLY SIMULATED]. No QPU hardware used. All AlphaFold comparison numbers sourced from published literature (DOI: 10.1002/prot.26257), not re-run.**

*Attributed to: Tommaso Marena, Independent Undergraduate Researcher. No funding. No external assistance. Full rigor.*
