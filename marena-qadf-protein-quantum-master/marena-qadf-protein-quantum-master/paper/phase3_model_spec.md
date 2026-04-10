# Phase 3 — Model Architecture Specification
## Hybrid Quantum-Classical Protein Structure Prediction (QADF Project)
### Target: Side-Chain Rotamer Classification with Calibrated Confidence

---

## A. Input Representation

### A.1 Primary Sequence Features

For a protein of n residues, the input consists of:

**One-hot amino acid encoding**:
- 20-dimensional one-hot vector per residue (standard amino acid alphabet)
- Dimension: n × 20

**Physicochemical features** (per residue, 9 features):
| Feature | Description | Range |
|---|---|---|
| Hydrophobicity | Kyte-Doolittle scale, normalized | [−1, 1] |
| Charge | Net formal charge at pH 7 | {−1, 0, +1} |
| Molecular weight | Residue MW, normalized by max | [0, 1] |
| Solvent accessibility | Predicted RSA (relative) | [0, 1] |
| Secondary structure | 3-hot: helix / sheet / coil | {0,1}³ |
| B-factor (if known) | From PDB, normalized | [0, 1] |

Total feature vector per residue: 20 + 9 = **29 dimensions**

**Pairwise distance matrix**:
- Cα–Cα Euclidean distance matrix D ∈ ℝⁿˣⁿ
- Normalized by max observed distance in the structure
- Used to construct k-nearest-neighbor graph (k=5) for EGNN message passing

### A.2 Rotamer Priors

**Dunbrack backbone-dependent rotamer library** [REF-05, DOI: 10.1002/prot.22921]:
- Prior probability p(rotamer | φ, ψ, amino acid) for each residue
- Discretized to 3 bins per chi angle: g− (−60°), t (180°), g+ (+60°)
- Encoded as per-residue prior probability vector (length = number of rotamer states)
- Residues with no chi angles (Gly, Ala) receive a dummy singleton state
- Input dimension contribution: n × max_rotamer_states (padded to 9 for χ₁×χ₂)

**Ramachandran priors**:
- Per-residue (φ, ψ) dihedral angle (from PDB backbone, measured in degrees)
- Converted to (sin φ, cos φ, sin ψ, cos ψ) — 4D periodic encoding per residue
- Avoids wrap-around discontinuity at ±180°

---

## B. Classical Backbone: Equivariant Graph Neural Network (EGNN)

### B.1 Justification

The backbone of the model is a small **Equivariant Graph Neural Network (EGNN)**. Equivariant architectures are appropriate for molecular data because protein structures are naturally SE(3)-invariant: the correct rotamer assignment does not change under global rotation or translation of the structure. EGNN maintains this invariance by updating both node features and 3D coordinates through equivariant message-passing layers [REF-15, Genome Biology 2025, PMC12665208].

The ENGINE paper [REF-15] demonstrates that EGCL (equivariant graph convolutional layer) updates:
- Node features: hᵢ → hᵢ' via messages aggregated from neighbors
- Coordinates: xᵢ → xᵢ' via radial component of position differences

This guarantees that the model is invariant to global rotations/translations and equivariant to permutation of atoms — essential for structural inputs.

### B.2 Architecture

```
Input Layer:
  Per-residue features: [one-hot(20) | physicochemical(9) | rotamer_prior(9) | 
                         rama_encoding(4)] → 42-dim per residue

Node embedding:
  Linear(42 → 32) + LayerNorm + ReLU

3D coordinate input: Cα positions (x, y, z) ∈ ℝ³

EGNN Layer 1:
  Message passing on k=5 NN graph
  Hidden dim: 32
  Edge features: relative distance, normalized (scalar)
  EGCL update: (hᵢ, xᵢ) → (hᵢ', xᵢ')
  
EGNN Layer 2:
  Same architecture
  Output: node embeddings hᵢ ∈ ℝ³²

Global pooling (for latent quantum embedding):
  Mean pooling over 4-residue local window → 32-dim context vector
```

**Why 2 layers**: Given structures of ≤25 residues with k=5 nearest neighbors, 2 message-passing layers allows each residue to aggregate information from all residues within 2 hops — sufficient for local chemical context. Deeper networks risk overfitting on small datasets.

**Why hidden dim 32**: Balanced between expressive power and computational cost for small PDB structures. At n=20 residues and batch=8, memory footprint is negligible. Matches scale used in proof-of-concept molecular GNNs for NISQ-era quantum-classical hybrids [REF-12, arXiv: 2502.11951].

---

## C. Quantum Module: Parameterized Quantum Circuit

### C.1 Architecture

**Platform**: PennyLane `default.qubit` (classical simulation) [CLASSICALLY SIMULATED]
**Qubit count**: 8 qubits (fixed-size input; mean-pooled EGNN embedding projected down)

**Circuit structure (angle embedding + variational layers)**:

```
Classical → Quantum Interface:
  Linear(32 → 8) → tanh → 8 angle parameters θ ∈ [−π, π]

Quantum Circuit [CLASSICALLY SIMULATED]:
  Layer 0: AngleEmbedding(θ, wires=[0..7], rotation='Y')
  
  Layer 1 (Variational, p=1):
    CNOT ladder: CNOT(0,1), CNOT(1,2), ..., CNOT(6,7)
    Ry(φ₁ᵢ) on each qubit i
  
  Layer 2 (Variational, p=2):
    CNOT ladder (reversed): CNOT(7,6), ..., CNOT(1,0)
    Rz(φ₂ᵢ) on each qubit i
  
  Measurement: ⟨Zᵢ⟩ expectation values, i=0..7
  Output: 8-dim vector ∈ [−1, +1]

Quantum → Classical Interface:
  Linear(8 → 32) → ReLU
```

**Total trainable parameters in quantum module**:
- 8 (input embedding) + 16 (Ry gates layer 1) + 16 (Rz gates layer 2) = **40 quantum parameters**
- Plus 2 linear layers (32×8 + 8×32 = 512 classical parameters)

### C.2 Justification

The parameterized quantum circuit (PQC) acts as a **latent feature transformer** on the fixed-size 32-dimensional EGNN embedding. The role of the quantum module is to perform a quantum-native transformation of the classical embedding before the output head decodes rotamer classes and confidence scores.

**Falsifiable scientific claim** [CLASSICALLY SIMULATED]:

> Under a fixed compute budget (equal number of floating-point parameters and operations), the quantum feature transformation (8-qubit PQC with 40 trainable parameters) produces lower average side-chain energy solutions than an equivalent classical MLP layer (Linear(32→8→32), ~352 parameters) on the QUBO instances tested, as measured by the mean QUBO objective value over the 1L2Y test residues.

This claim is falsifiable because:
1. The MLP and PQC have comparable parameter counts (~40 vs ~352 — intentionally making the quantum module smaller to test efficiency per parameter, not total capacity)
2. Results are evaluated on real PDB data (1L2Y), not synthetic benchmarks
3. The comparison metric (QUBO objective value) is independently computable
4. All quantum results are classically simulated with PennyLane `default.qubit` and labeled [CLASSICALLY SIMULATED]

**Why quantum feature transformation explores rotamer superposition**:
The AngleEmbedding encodes the classical residue embedding as rotation angles on qubits. Under CNOT entanglement, the circuit creates a superposition of rotamer-relevant quantum states. The variational layers (Ry, Rz) learn to amplify states corresponding to low-energy rotamer configurations. While classical simulation does not achieve quantum speedup, the functional form of the PQC is distinct from classical MLPs — it computes trigonometric functions of sums/differences of parameters in ways that are difficult to replicate with a shallow MLP. This is the basis for the claim of complementary representational power (not speedup) at classical simulation scale.

---

## D. Output Head

### D.1 Rotamer Class Probabilities

For each residue rᵢ:
- Project combined embedding (EGNN + quantum module): Linear(64 → num_rotamer_states_i)
- Softmax → per-residue rotamer class probability distribution p(sᵢ | rᵢ)
- **num_rotamer_states** varies by amino acid (1 for Gly/Ala; up to 9 for Arg, Lys)
- For simplicity in the 3-bin encoding: num_rotamer_states = 3 for all residues with χ₁

**Prediction**: argmax p(sᵢ | rᵢ) → predicted rotamer bin → predicted mean χ₁ angle

### D.2 Per-Residue Confidence Score

Inspired by AlphaFold 2's pLDDT [REF-04] but calibrated [REF-08]:

- Project combined embedding: Linear(64 → 1) → Sigmoid → confidence ∈ [0, 1]
- Scale to [0, 100]: confidence_score = 100 × sigmoid(logit)
- Calibrated to approximate P(χ₁ error < 40° | confidence_score)
- Color-coded per AlphaFold convention [REF-13]:
  - >90: dark blue (#0053D6) — Very high confidence
  - 70–90: light blue (#65CBF3) — Confident
  - 50–70: yellow (#FFDB13) — Low confidence
  - <50: orange (#FF7D45) — Very low confidence

**Difference from pLDDT**: pLDDT regresses per-residue lDDT-Cα (backbone metric) and is NOT a calibrated probability [REF-08]. Our confidence score regresses |predicted_χ₁ − actual_χ₁| ≤ 40° (a side-chain accuracy metric), and is explicitly calibrated using a calibration regularizer in the loss function.

---

## E. Loss Function

The total loss combines five terms:

```
L_total = L_CE + λ₁ L_clash + λ₂ L_torsion + λ₃ L_calib + λ₄ L_dunbrack
```

### E.1 Cross-entropy on Rotamer Classes

```
L_CE = −(1/n) Σᵢ Σⱼ y_{ij} log p(sᵢ = j)
```
where y_{ij} = 1 if residue i's ground-truth rotamer is class j.

### E.2 Steric Clash Penalty

Penalizes van der Waals overlaps between predicted rotamer conformations:
```
L_clash = Σᵢ Σⱼ≠ᵢ max(0, r_vdw_ij − d_ij(predicted))²
```
where d_ij is the distance between predicted side-chain heavy atoms and r_vdw_ij is the sum of van der Waals radii. Computed differentiably using predicted mean chi angles (not discrete bins). **λ₁ = 0.1**

### E.3 Torsion Periodicity Regularizer

Penalizes predicted chi angle means that violate known periodicity constraints:
```
L_torsion = Σᵢ (1 − cos(χ₁_pred_i − χ₁_dunbrack_mode_i))²
```
where χ₁_dunbrack_mode_i is the modal chi angle from the Dunbrack library [REF-05] given the backbone φ/ψ. **λ₂ = 0.05**

### E.4 Calibration Regularizer

Penalizes miscalibration between predicted confidence and observed accuracy:
```
L_calib = Σ_bins |acc(bin) − conf(bin)|
```
where bins are 10 equal-width confidence intervals and acc/conf are empirical accuracy and mean confidence within each bin. This is the differentiable approximation of the Expected Calibration Error (ECE), motivated by CalPro [REF-08]. **λ₃ = 0.2**

### E.5 Dunbrack Prior Regularizer

KL divergence between predicted rotamer distribution and Dunbrack prior:
```
L_dunbrack = Σᵢ KL(p_predicted_i || p_dunbrack_i)
```
Ensures predictions do not deviate wildly from known rotamer statistics. **λ₄ = 0.1**

---

## F. Training Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate; standard for GNN training |
| Learning rate | 1e-3 | Standard initial LR for Adam on protein models |
| Batch size | 8 | Appropriate for ≤25-residue proteins; GPU memory efficient |
| Max epochs | 500 | With early stopping |
| Early stopping patience | 20 | Stop if validation loss does not improve for 20 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | |
| Weight initialization | Xavier uniform | |
| Random seeds | 42, 123, 456 | Three seeds for reproducibility and variance estimation |
| Gradient clipping | max_norm = 1.0 | Prevent gradient explosion in EGNN layers |
| Validation metric | χ₁ mean absolute error (MAE) | Primary metric; less sensitive to class imbalance than accuracy |

**Hardware target**: Single GPU or CPU (structures ≤25 residues; batch=8 is fast on CPU). Expected training time: <30 minutes for 500 epochs on CPU.

---

## G. Ablation Plan (6 Conditions)

| Condition | Description | Purpose |
|---|---|---|
| **G1: Full model** | EGNN + Quantum PQC + Calibration loss | Baseline full model |
| **G2: No quantum module** | EGNN + Classical MLP(32→8→32) replaces PQC | Tests quantum contribution |
| **G3: No EGNN** | Classical MLP backbone + Quantum PQC | Tests EGNN contribution |
| **G4: No calibration loss** | Full model without L_calib (λ₃=0) | Tests calibration regularizer effect |
| **G5: No Dunbrack prior** | Full model without L_dunbrack and rotamer priors | Tests prior knowledge contribution |
| **G6: EGNN only** | EGNN + MLP head, no quantum, no calibration loss | Minimal classical baseline |

**Primary comparison**: G1 vs. G2 (quantum vs. classical feature transformer with matched architecture)

**Metrics for all conditions** (evaluated on held-out test set, 3 seeds each):
- χ₁ MAE (degrees) — primary metric
- χ₂ MAE (degrees, where applicable)
- Rotamer accuracy (fraction of residues with correct bin assignment)
- Mean Calibration Error (ECE)
- Steric clash frequency (fraction of residue pairs with overlap > 0.1 Å)
- QUBO objective value for test window (direct comparison to exhaustive search)

**Statistical testing**: Paired Wilcoxon signed-rank test between G1 and G2 on per-residue χ₁ MAE (Phase 7).

---

## H. Architecture Diagram (ASCII)

```
INPUT PER RESIDUE:
  Sequence + Physicochemical (29-dim)
  Rotamer prior (9-dim)
  Ramachandran angles → (sin/cos φ, sin/cos ψ) (4-dim)
  Cα coordinates (3-dim) → used for graph construction
                                    │
                                    ▼
                         ┌─────────────────┐
                         │  Node Embedding  │
                         │  Linear(42→32)   │
                         │  LayerNorm + ReLU│
                         └────────┬────────┘
                                  │ node features h, coords x
                                  ▼
                         ┌─────────────────┐
                         │   EGNN Layer 1   │
                         │  Message passing │
                         │  (k=5 NN graph) │
                         │  hidden dim: 32  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   EGNN Layer 2   │
                         │  Message passing │
                         │  hidden dim: 32  │
                         └────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
              LOCAL WINDOW                OUTPUT DIRECT
              (mean pool n=4)                  │
                     │                         │
                     ▼                         │
           ┌──────────────────┐                │
           │  Linear(32 → 8)  │                │
           │  tanh             │                │
           └────────┬─────────┘                │
                    │ 8 angle params            │
                    ▼                           │
          ┌──────────────────────┐             │
          │  QUANTUM PQC (8 qb)  │             │
          │  [CLASSICALLY SIM.]  │             │
          │  AngleEmbedding      │             │
          │  CNOT ladder         │             │
          │  Ry, Rz rotations    │             │
          │  ⟨Zᵢ⟩ measurements   │             │
          └────────┬─────────────┘             │
                   │ 8-dim output              │
                   ▼                           │
          ┌──────────────────┐                │
          │  Linear(8 → 32)  │                │
          │  ReLU            │                │
          └────────┬─────────┘                │
                   │                          │
                   └──────────┬───────────────┘
                              │ concat → 64-dim
                              ▼
                   ┌──────────────────────┐
                   │    OUTPUT HEAD        │
                   ├──────────────────────┤
                   │  Rotamer branch:      │
                   │  Linear(64 → 3)       │
                   │  Softmax → p(sᵢ)      │
                   │  → χ₁ class (g-,t,g+) │
                   ├──────────────────────┤
                   │  Confidence branch:   │
                   │  Linear(64 → 1)       │
                   │  Sigmoid × 100        │
                   │  → confidence ∈[0,100]│
                   └──────────────────────┘
                              │
                              ▼
                    LOSS FUNCTION:
                    L_CE + λ₁ L_clash + λ₂ L_torsion
                    + λ₃ L_calib + λ₄ L_dunbrack
```

---

*References: REF-02 (arXiv: 2507.19383), REF-04 (DOI: 10.1002/prot.26257), REF-05 (DOI: 10.1002/prot.22921), REF-08 (arXiv: 2601.07201), REF-12 (arXiv: 2502.11951), REF-13 (AlphaFold pLDDT convention), REF-15 (ENGINE, Genome Biology 2025, PMC12665208)*
