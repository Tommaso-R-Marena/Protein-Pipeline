# QuantumProteinOS

[![CI](https://github.com/example/QuantumProteinOS/actions/workflows/ci.yml/badge.svg)](https://github.com/example/QuantumProteinOS/actions/workflows/ci.yml)
[![Full Benchmark Evaluation](https://github.com/example/QuantumProteinOS/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/example/QuantumProteinOS/actions/workflows/benchmarks.yml)


A unified quantum-classical research framework for protein structure prediction, addressing three principal failure modes of AlphaFold 3:
1. Conformational diversity (fold-switching, autoinhibition)
2. Intrinsic disorder prediction
3. Stereochemical correctness (chirality for L/D proteins)

## Quick Start
```bash
# 1. Install dependencies
git clone https://github.com/example/QuantumProteinOS.git
cd QuantumProteinOS
pip install -r requirements.txt
pip install -e .

# 2. Predict from sequence
python scripts/run_full_pipeline.py --sequence MQIFVKTLTGKTITLEVEPS --output results/

# 3. Predict from PDB
python scripts/run_full_pipeline.py --pdb protein.pdb --output results/

# 4. Correct AF3 output
python scripts/run_full_pipeline.py --af3-output af3_prediction.pdb --correct-chirality

# 5. Run full benchmark
python scripts/run_full_pipeline.py --benchmark all --output benchmarks/results/
```

## Benchmarks
| Benchmark | Target Metric | Target Value | vs AF3 / Baseline |
|-----------|---------------|--------------|-------------------|
| DisorderNet (DisProt) | AUC-ROC | 0.831 | AF3 0.747 |
| ChiralFold (Validation) | D-residues corrected | 100% | AF3 51% |
| QICESS Autoinhibited | imfdRMSD < 3Å | 54% | AF3 33% |
| IWS-QAOA N=4 SK-glass | Energy gap vs Exact | 0.0% | Greedy 8.1% |
| QCFold Fold-switching | Success rate | 60% | AF3 7.6% |

## Honest Limitations
This framework explores hybrid quantum-classical optimization (QAOA, VQE, QUBO mapping).
- **No QPU usage:** All quantum circuits are classically simulated via PennyLane `default.qubit` to ensure exact statevector tracking without hardware noise.
- **VQE Performance:** VQE does not systematically outperform random ranking at ≤16 qubits.
- **IWS-QAOA:** Displays an advantage over *single-shot* greedy algorithms on SK-glass frustrated rotational geometries, but classical Simulated Annealing (SA) remains broadly optimal with sufficient iterations. Quantum advantage at >50 qubits remains theoretical.

## Architecture
```text
QuantumProteinOS
 ├── qpos/disorder       # Intrinsic Disorder (DisorderNet)
 ├── qpos/chirality      # Stereochemical Auditing (ChiralFold, ChiralBoltz)
 ├── qpos/conformational # Conformational Sampling (QICESS, QuantumFoldX)
 ├── qpos/quantum        # Optimizers (qprotein-iws, QADF)
 └── qpos/rotamers       # Rotamer packing & sidechain modeling
```

## Sources
- marena-qadf-protein-quantum
- marena-qadf-v2-scaled
- QuantumFoldBench
- DisorderNet
- QuantumFoldX
- QCFold
- ChiralBoltz
- qprotein-iws
- ChiralFold
