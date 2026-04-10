#!/usr/bin/env python
"""
Evaluate chirality violation rate on the Childs et al. (2025) benchmark.

Compares:
  1. Baseline Boltz-2 (no mirror aug)
  2. ChiralBoltz (mirror aug fine-tuned)
  3. AlphaFold 3 reference result: 51% violation rate (Childs et al. 2025)

Usage:
    python scripts/evaluate_chirality.py \
        --checkpoint outputs/chiralboltz/checkpoints/best.ckpt \
        --n_seeds 5
"""
import argparse
from chiralboltz.data.validation_set import get_benchmark_systems
from chiralboltz.evaluate.chirality_metrics import chirality_violation_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_seeds",    type=int, default=5)
    args = parser.parse_args()

    systems = get_benchmark_systems()
    print(f"Benchmark: {len(systems)} systems (Childs et al. 2025)")
    print(f"AlphaFold 3 baseline violation rate: 51% (3,255 samples)")
    print()

    for sys in systems:
        print(f"  {sys['name']} ({sys['pdb_id']}) — type: {sys['type']}, reflect: {sys['reflect']}")

    print("\nTo run evaluation: load model checkpoint, generate predictions with n_seeds,")
    print("then call chirality_violation_rate() on predicted vs. crystal coords.")


if __name__ == "__main__":
    main()
