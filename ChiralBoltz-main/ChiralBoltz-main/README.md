# ChiralBoltz: Mirror-Image Augmentation for Heterochiral Complex Structure Prediction

**ChiralBoltz** is a mirror-image augmentation fine-tuning pipeline built on top of [Boltz-2](https://github.com/jwohlwend/boltz) (MIT license). It enables Boltz-2 to correctly predict **heterochiral complexes** — D-peptide:L-protein interfaces, D-protein apo structures, and other systems where standard structure prediction models fail.

## Motivation

AlphaFold 3 produces structures with a **51% chirality violation rate** across 3,255 experiments on heterochiral targets ([Childs et al. 2025](https://doi.org/10.1101/2025.03.14.643307)) — essentially random chance. This happens because:

1. Training data is overwhelmingly L-protein (>99.9% of the PDB)
2. The model learns L-stereochemistry as a prior, not a physical constraint
3. D-amino acids are treated as L-amino acids with wrong coordinates

ChiralBoltz solves this by exploiting a key symmetry: **every L-protein structure in the PDB can be reflected to generate a valid D-protein training example**, making training data effectively unlimited.

## Method Overview

```
                        ┌─────────────────────────┐
                        │     PDB L-structures     │
                        │   (100,000+ examples)    │
                        └───────────┬─────────────┘
                                    │
                           reflect x → -x
                           flip chirality flags
                                    │
                        ┌───────────▼─────────────┐
                        │   Mirror D-structures    │
                        │   (100,000+ examples)    │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
          ┌─────────▼─────────┐           ┌─────────▼─────────┐
          │  L-protein batch  │           │  D-protein batch  │
          │   (original)      │           │   (mirrored)      │
          └─────────┬─────────┘           └─────────┬─────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │      Boltz-2 Model      │
                        │   (pre-trained weights)  │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
          ┌─────────▼─────────┐           ┌─────────▼─────────┐
          │   L_diffusion     │           │   L_chiral        │
          │   (Boltz-2 base)  │           │   (chiral volume) │
          └─────────┬─────────┘           └─────────┬─────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                          L_total = L_diffusion
                            + λ * L_chiral
```

### Key Components

1. **Mirror-Image Augmentation** (`chiralboltz/augmentation/mirror.py`): Reflects all atom coordinates through the y-z plane (`x → -x`) and inverts CW/CCW chirality flags. This converts any L-protein to a physically valid D-protein.

2. **Differentiable Chiral Volume Loss** (`chiralboltz/loss/chiral_volume.py`): Computes the signed scalar triple product at each chiral center and penalizes sign disagreements between predicted and reference structures using a hinge loss formulation.

3. **MirrorAugmentedDataset** (`chiralboltz/augmentation/dataset.py`): Wraps any Boltz-2 dataset and applies stochastic mirror augmentation with configurable probability (default 50%).

4. **ChiralBoltzModule** (`chiralboltz/training/trainer.py`): PyTorch Lightning module that fine-tunes Boltz-2 with the auxiliary chiral volume loss, including linear warmup scheduling.

## Installation

```bash
# Clone the repository
git clone https://github.com/Tommaso-R-Marena/ChiralBoltz.git
cd ChiralBoltz

# Install with Boltz-2 dependency
pip install -e .

# For development
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- [Boltz-2](https://github.com/jwohlwend/boltz) >= 0.4.0
- PyTorch Lightning >= 2.2.0

## Quick Start

### 1. Prepare Mirror Dataset

Generate mirrored D-protein structures from your PDB NPZ files:

```bash
python scripts/prepare_mirror_dataset.py \
    --source_dir /path/to/rcsb_processed_targets/structures \
    --output_dir /path/to/mirror_structures \
    --n_workers 8
```

### 2. Fine-Tune Boltz-2

```bash
# Edit the config with your data paths
vim configs/chiralboltz_finetune.yaml

# Launch fine-tuning
python scripts/fine_tune.py --config configs/chiralboltz_finetune.yaml
```

### 3. Evaluate Chirality

```bash
python scripts/evaluate_chirality.py \
    --checkpoint outputs/chiralboltz/checkpoints/best.ckpt \
    --n_seeds 5
```

## Connection to ChiralFold

ChiralBoltz uses the signed chiral volume validation approach from [ChiralFold](https://github.com/Tommaso-R-Marena/ChiralFold) (Marena 2026). The `chirality_violation_rate` metric directly corresponds to the ChiralFold validator, enabling consistent comparison across models.

The chiral volume loss in ChiralBoltz is a differentiable version of the same geometric test:

```
V_i = (r_center - r_j) · ((r_center - r_k) × (r_center - r_l))
```

A correct chiral center has `sign(V_pred) == sign(V_ref)`. ChiralBoltz trains the model to maintain this invariant through a soft hinge loss.

## Project Structure

```
ChiralBoltz/
├── chiralboltz/
│   ├── augmentation/       # Mirror-image transforms and dataset wrapper
│   ├── loss/               # Differentiable chiral volume loss
│   ├── training/           # Lightning module and config
│   ├── data/               # PDB preprocessing and benchmark loaders
│   └── evaluate/           # Chirality violation metrics
├── scripts/                # CLI entry points
├── configs/                # YAML training/eval configs
└── tests/                  # Unit tests
```

## Citations

If you use ChiralBoltz, please cite the following:

```bibtex
@article{childs2025,
  title={Has AlphaFold 3 Solved the Protein Folding Problem for D-Peptides?},
  author={Childs, Corey M. and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.14.643307}
}

@article{ishitani2025,
  title={Signed Chiral Volume for Stereochemical Validation of Protein Structures},
  author={Ishitani, Rintaro and Moriwaki, Yoshitaka},
  journal={ACS Omega},
  year={2025}
}

@software{wohlwend2024boltz,
  title={Boltz-2: Biomolecular Structure Prediction},
  author={Wohlwend, Jeremy and others},
  url={https://github.com/jwohlwend/boltz},
  year={2024}
}

@software{marena2026chiralfold,
  title={ChiralFold: Signed Chiral Volume Validator for Structure Predictions},
  author={Marena, Tommaso R.},
  url={https://github.com/Tommaso-R-Marena/ChiralFold},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
