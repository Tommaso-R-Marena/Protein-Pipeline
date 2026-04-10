# Claims Registry

This document tracks all claims with evidence ratings. Every claim must have an empirical source, and honest limitations must be explicitly declared to prevent overstating NISQ performance.

| Claim | Evidence Rating | Source | Caveat |
|-------|----------------|--------|--------|
| DisorderNet v6 AUC = 0.831 | Strongly supported | DisProt full benchmark | CPU model only |
| ChiralFold 0% chirality violations | Strongly supported | Childs et al. 2025 benchmark | D-peptide geometry construction |
| ChiralFold 29 D/L mismatches | Strongly supported | PDB-wide manual validation | Out of 4,616 structures |
| QICESS 54% imfdRMSD < 3 Å | Strongly supported | 22-protein autoinhibited benchmark | vs AF3 33%; only vs AF2 survives at α=0.01 |
| IWS-QAOA 0.0% gap on SK-glass N=4 | Classically Simulated | Exact statevector simulation | 12 qubits; no QPU |
| IWS-QAOA beats single-shot greedy | Classically Simulated | N=4/5 SK-glass frustration | Not against Iterated Local Search or SA |
| VQE no advantage at 16 qubits | Strongly supported | QuantumFoldX ablation p=0.25 | 14 proteins only |
| QCFold 60% success on fold-switch | Weakly supported | 5-protein synthetic demo | Coordinate proof-of-concept only |
