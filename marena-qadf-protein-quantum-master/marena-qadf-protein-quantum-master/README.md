# Quantum Amenability Decision Framework (QADF) for Protein Side-Chain Rotamer Optimization

## Description

This project develops and demonstrates the **Quantum Amenability Decision Framework (QADF)**, a structured rubric for evaluating whether a given computational subproblem in structural biology is a viable near-term target for quantum optimization. As a concrete proof-of-concept, the framework is applied to **side-chain rotamer optimization** — the problem of assigning low-energy rotamer states to amino acid side chains given a fixed backbone — for two benchmark PDB structures: 1L2Y (Trp-cage, 20 residues) and 1UBQ (ubiquitin, 76 residues). The subproblem is encoded as a Quadratic Unconstrained Binary Optimization (QUBO) problem using a sliding 4-residue window with 12 binary variables per window and one-hot encoding of rotamer states. Classical baselines (exhaustive search, greedy assignment, simulated annealing) and classically simulated quantum experiments (QAOA at p=1 and p=2 via PennyLane `default.qubit`) are compared on solution quality and computational cost. The project further develops a hybrid ML model that generates per-residue confidence scores analogous to AlphaFold's pLDDT, and calibrates those scores against experimentally observable disorder proxies. The central scientific result is not that QAOA outperforms classical methods at this scale — it does not, and this is reported honestly — but that the QADF rubric provides a reproducible, multi-dimensional scoring system for assessing quantum readiness, and that rotamer optimization scores favorably on this rubric relative to global backbone prediction.

**Author:** Tommaso Marena, undergraduate student, Departments of Chemistry and Philosophy, The Catholic University of America. Independent research; no external funding or institutional affiliation with this project.

> **Disclaimer:** All quantum results in this project are classically simulated via PennyLane `default.qubit`. No QPU hardware was used. Reported quantum circuit results represent ideal noiseless simulation and, where noted, depolarizing noise injection at ε = 0.01. Scaling beyond ~20–25 qubits is not attempted; this ceiling is a feature, not a bug, of honest classical simulation.

---

## Repository Structure

```
marena-qadf/
│
├── README.md                   # This file
├── MEMO.md                     # Advisor-facing research memo
├── CLAIMS.md                   # Claims registry with evidence ratings
├── VENUES.md                   # Publication and presentation venue assessment
├── requirements.txt            # Pinned Python dependencies
│
├── data/
│   ├── setup_and_data.py       # Downloads PDB structures, extracts rotamer data
│   ├── qubo_encoding.py        # Builds QUBO matrices from rotamer energy terms
│   ├── pdb/                    # Downloaded PDB files (1L2Y.pdb, 1UBQ.pdb)
│   ├── rotamers/               # Extracted rotamer states and Dunbrack probabilities
│   └── qubo_matrices/          # Serialized QUBO matrices (.npy format)
│
├── models/
│   ├── gnn_model.py            # Equivariant GNN for per-residue confidence scoring
│   ├── confidence_head.py      # Calibration head (Platt scaling / temperature)
│   └── checkpoints/            # Saved model weights (if training is run)
│
├── results/
│   ├── experiments.py          # Runs all classical and simulated quantum experiments
│   ├── confidence_analysis.py  # Confidence calibration and ECE computation
│   ├── visualization.py        # Generates all figures
│   ├── figures/                # Output figures (PDF and PNG)
│   └── benchmarks/             # Numerical results tables (CSV)
│
└── tests/
    ├── test_qubo.py            # Unit tests for QUBO construction
    ├── test_qaoa.py            # Unit tests for QAOA circuit construction
    └── test_calibration.py     # Unit tests for calibration pipeline
```

---

## Environment Setup

**Python version:** 3.10 or higher is required.

### Dependencies

```
pennylane>=0.38
biopython>=1.83
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
seaborn>=0.13
torch>=2.2
pandas>=2.2
networkx>=3.2
scikit-learn>=1.4
qiskit-aer>=0.13
```

> **Notes:**
> - `torch>=2.2` is used for the GNN model. If GPU resources are unavailable or undesired, `scikit-learn>=1.4` provides a fallback for the confidence estimation head (linear calibration only).
> - `qiskit-aer>=0.13` is listed as a fallback simulator; all primary quantum experiments use PennyLane `default.qubit`.
> - `pennylane>=0.38` is required for the `qml.QAOA` module interface and gradient-based optimization used in the p=1 and p=2 experiments.

### Installation

```bash
pip install -r requirements.txt
```

It is strongly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduction Steps

The following steps take a user from a clean environment to a full reproduction of all figures, tables, and numerical results reported in the paper.

**1. Clone or download the repository.**

```bash
git clone https://github.com/[username]/marena-qadf.git
cd marena-qadf
```

If you received this as a ZIP archive, unzip it and navigate into the directory.

**2. Create and activate a virtual environment.**

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

**3. Install all dependencies.**

```bash
pip install -r requirements.txt
```

Verify PennyLane is correctly installed:

```bash
python -c "import pennylane as qml; print(qml.__version__)"
```

Expected output: `0.38.x` or higher.

**4. Run `data/setup_and_data.py` — download PDB structures and extract rotamers.**

```bash
python data/setup_and_data.py
```

This script:
- Downloads `1L2Y.pdb` and `1UBQ.pdb` from the RCSB PDB via Biopython's `PDBList` interface.
- Parses backbone and side-chain coordinates.
- Assigns each residue to a Dunbrack rotamer bin (χ1/χ2 discretization at 60° resolution).
- Computes pairwise interaction energies for the 4-residue sliding window.
- Saves processed data to `data/rotamers/` and `data/pdb/`.

Expected runtime: 1–3 minutes (network-dependent for download).

**5. Run `data/qubo_encoding.py` — build QUBO matrices.**

```bash
python data/qubo_encoding.py
```

This script:
- Reads the extracted rotamer interaction energies from `data/rotamers/`.
- Constructs QUBO matrices for each 4-residue window using one-hot encoding (12 binary variables per window, assuming 3 rotamer states per residue).
- Applies one-hot penalty terms (λ = 5.0) to enforce valid rotamer assignments.
- Saves QUBO matrices as `.npy` files to `data/qubo_matrices/`.

**6. Run `results/experiments.py` — run all classical and simulated quantum experiments.**

```bash
python results/experiments.py
```

This script runs, for each 4-residue window across both PDB structures:
- **Exhaustive search** (exact ground truth, feasible at 12 qubits).
- **Greedy assignment** (O(N) baseline).
- **Simulated annealing** (scipy-based, 10 random restarts).
- **QAOA p=1** (PennyLane `default.qubit`, COBYLA optimizer, 50 iterations).
- **QAOA p=2** (PennyLane `default.qubit`, COBYLA optimizer, 100 iterations).
- **Depolarizing noise experiment** (QAOA p=1 with ε=0.01 applied after each gate layer, 10 shots averaged).

Results are written to `results/benchmarks/` as CSV files. **Expected runtime: 30–90 minutes** depending on hardware, due to repeated statevector simulation over all windows.

**7. Run `results/confidence_analysis.py` — compute and calibrate confidence scores.**

```bash
python results/confidence_analysis.py
```

This script:
- Loads GNN model predictions (or computes from scratch if no checkpoint exists).
- Computes per-residue confidence scores.
- Calibrates against B-factor-derived disorder proxy using Platt scaling.
- Computes Expected Calibration Error (ECE) before and after calibration.
- Saves calibration curves and ECE values to `results/benchmarks/calibration_results.csv`.

**8. Run `results/visualization.py` — generate all figures.**

```bash
python results/visualization.py
```

Generates all figures referenced in the paper, saved to `results/figures/` as both PDF (publication quality) and PNG (web display). Figures include:
- QADF scoring radar chart for 8 subproblems across 9 dimensions.
- Energy landscape comparison (exhaustive, greedy, SA, QAOA p=1, QAOA p=2).
- Scaling cost vs. qubit count.
- Confidence calibration curves (pre- and post-calibration).
- Noise sensitivity of QAOA solution quality.

**9. Find outputs.**

- Figures: `results/figures/`
- Numerical tables: `results/benchmarks/`
- Calibration analysis: `results/benchmarks/calibration_results.csv`

---

## Key Findings Summary

This project demonstrates that:

1. **The QADF rubric is applicable and discriminating.** Applied to 8 structural biology subproblems across 9 scoring dimensions (qubit count, locality, connectivity, noise tolerance, available encodings, classical hardness, problem size scaling, biological relevance, and data availability), the rubric produces a ranked ordering that is consistent with the current state of the quantum optimization literature. Rotamer optimization scores in the top tier; backbone folding scores near the bottom.

2. **QAOA does not outperform greedy or simulated annealing at the 12-qubit scale studied here.** This is an expected and honest result. The instances are small enough that classical methods find exact or near-exact solutions trivially. The comparison is included to establish a clean baseline and to demonstrate that the quantum experiments are well-implemented, not to claim quantum advantage.

3. **Classical simulation imposes a hard ceiling at approximately 20–25 qubits** under repeated experimental conditions with the hardware available for this project. This ceiling is documented and reported as a constraint, not overcome.

4. **Per-residue confidence calibration is achievable.** After Platt scaling against B-factor-derived disorder proxies, the model's ECE improves meaningfully, suggesting that the calibration approach is technically sound even if the underlying model's raw accuracy is limited by training data size.

5. **The primary contribution is the QADF framework itself** — a reusable, multi-dimensional scoring tool for quantum readiness assessment that is grounded in the current literature and validated against known results.

---

## License

MIT License. See `LICENSE` for full text.

---

## Citation

If you use this project, the QADF rubric, or any component of this codebase in your own work, please cite:

```
Marena, T. (2025). Quantum Amenability Decision Framework (QADF) for Protein 
Side-Chain Rotamer Optimization. Independent research, The Catholic University 
of America. Available at: https://github.com/[username]/marena-qadf
```

### Key References

This project builds directly on and cites the following published works:

- Doga et al. (2024). DOI: [10.1021/acs.jctc.4c00067](https://doi.org/10.1021/acs.jctc.4c00067)
- Agathangelou et al. (2025). arXiv: [2507.19383](https://arxiv.org/abs/2507.19383)
- Khatami et al. (2023). DOI: [10.1371/journal.pcbi.1011033](https://doi.org/10.1371/journal.pcbi.1011033)
- Jumper et al. (2021, CASP14). DOI: [10.1002/prot.26257](https://doi.org/10.1002/prot.26257)
- Dunbrack (2011) Rotamer Library. PubMed: [21645855](https://pubmed.ncbi.nlm.nih.gov/21645855/)
- Bauza et al. (2023). DOI: [10.1038/s41534-023-00733-5](https://doi.org/10.1038/s41534-023-00733-5)
- FlowPacker (2024). bioRxiv: [2024.07.05.602280](https://doi.org/10.1101/2024.07.05.602280)
- CalPro (2026). arXiv: [2601.07201](https://arxiv.org/abs/2601.07201)
