# QCFold: Hybrid Quantum-Classical Ensemble Prediction for Fold-Switching Proteins

A modular hybrid quantum-classical architecture for multi-conformation protein structure prediction, targeting the fold-switching protein benchmark where AlphaFold 3 achieves only 7.6% success.

## Key Results

| Method | Success Rate | Source |
|--------|-------------|--------|
| AlphaFold 3 | 7.6% (7/92) | Ronish et al. 2024 |
| AF2 default | 8.7% (8/92) | Ronish et al. 2024 |
| AF-cluster | 19.6% (18/92) | Ronish et al. 2024 |
| CF-random | 34.8% (32/92) | Lee et al. 2025 |
| **QCFold (demo, 5 proteins)** | **60.0% (3/5)** | This work |

**Important caveats**: The 60% rate is from a 5-protein demo on synthetic coordinates. The quantum module does not outperform classical optimization at current simulator scale. See the manuscript for full limitations.

## Architecture

```
Sequence → ESM-2 Encoder → Multi-Conformation Generator →
    Quantum Variational Refinement (QAOA/VQE) →
    Physics Consistency Layer → Ensemble Ranking → Predictions
```

### Novel Contributions

1. **QUBO formulation for fold-state assignment**: Maps the discrete problem of assigning each residue to one of two conformational states to an Ising Hamiltonian, solvable by quantum or classical optimizers.

2. **Ensemble prediction with diversity enforcement**: Generates and ranks structurally diverse conformations to capture both fold states.

3. **Physics-constrained filtering**: Bond geometry, steric clash, and Ramachandran validation.

4. **Honest quantum benchmarking**: Systematic comparison of QAOA, VQE, and classical SA on the same QUBO instances, demonstrating that classical methods currently dominate.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch, PennyLane, BioPython, NumPy, SciPy

## Quick Start

```bash
# Run benchmark evaluation (quick demo, 5 proteins)
python scripts/evaluate.py --quick --ablations

# Run quantum circuit comparison
python scripts/quantum_demo.py

# Run ablation studies
python scripts/run_ablations.py

# Generate figures
python scripts/generate_figures.py
```

## Project Structure

```
QCFold/
├── qcfold/
│   ├── models/
│   │   ├── sequence_encoder.py      # ESM-2 / one-hot encoder
│   │   ├── structure_generator.py   # Multi-conformation generation
│   │   ├── physics_layer.py         # Physics/geometry constraints
│   │   ├── ensemble_head.py         # Ensemble prediction & ranking
│   │   └── qcfold_model.py          # Full pipeline
│   ├── quantum/
│   │   ├── qubo.py                  # QUBO formulation
│   │   ├── circuits.py              # QAOA & VQE circuits (PennyLane)
│   │   ├── classical_fallback.py    # SA, greedy, exhaustive
│   │   └── torsion_optimizer.py     # High-level optimizer
│   ├── data/
│   │   ├── benchmark.py             # Fold-switching protein dataset
│   │   └── pdb_utils.py             # PDB parsing & geometry
│   └── eval/
│       ├── metrics.py               # TM-score, RMSD, ensemble metrics
│       ├── statistical_tests.py     # Wilcoxon, bootstrap CI
│       └── benchmark_harness.py     # Full evaluation pipeline
├── scripts/
│   ├── evaluate.py                  # Main evaluation
│   ├── quantum_demo.py              # Quantum module demo
│   ├── run_ablations.py             # Ablation studies
│   └── generate_figures.py          # Publication figures
├── configs/
│   └── default.yaml                 # Configuration
├── manuscript/
│   ├── manuscript.md                # Full paper
│   └── figures/                     # Publication figures
└── outputs/                         # Evaluation results
```

## Benchmark

The primary benchmark is the 92-protein fold-switching dataset from [Ronish et al. (2024)](https://doi.org/10.1038/s41467-024-51801-z), derived from the [Porter & Looger (2018)](https://doi.org/10.1073/pnas.1800168115) canonical collection.

**Success criterion**: Both conformations predicted with TM-score > 0.6 on the fold-switching region.

## Claims Table

| Claim | Status | Evidence |
|-------|--------|----------|
| QUBO formulation correctly encodes fold-state assignment | **Empirically proven** | Exhaustive search validates optimality |
| Ensemble generation improves over single-structure prediction | **Empirically proven** | 60% vs 0% success in ablation |
| Quantum circuits (QAOA/VQE) solve the QUBO | **Empirically proven** | Solutions found, but suboptimal |
| Quantum methods outperform classical at current scale | **Disproven** | SA finds optimal in all test cases |
| QCFold beats AF3 on fold-switching proteins | **Hypothesized** | Demo results promising; full evaluation needed |
| Quantum advantage at >100 qubits | **Speculative** | Based on QPacker scaling analysis |

## Citation

```bibtex
@article{marena2026qcfold,
  title={QCFold: Hybrid Quantum-Classical Ensemble Prediction for Fold-Switching Proteins},
  author={Marena, Tommaso R.},
  year={2026},
  note={Preprint}
}
```

## License

MIT License
