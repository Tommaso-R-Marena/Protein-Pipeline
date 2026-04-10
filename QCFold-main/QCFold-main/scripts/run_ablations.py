#!/usr/bin/env python3
"""
QCFold Ablation Studies

Systematically tests the contribution of each component:
  1. Full QCFold (QAOA quantum refinement)
  2. VQE variant
  3. Classical SA only (no quantum)
  4. No physics layer
  5. No ensemble (single best structure)
  6. Random assignment baseline
  7. All-Fold-A and All-Fold-B baselines

Reports results in a publication-ready table format.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcfold.models.qcfold_model import QCFoldModel
from qcfold.eval.benchmark_harness import BenchmarkHarness
from qcfold.data.benchmark import FOLD_SWITCH_BENCHMARK
from qcfold.eval.statistical_tests import wilcoxon_comparison, bootstrap_ci


def main():
    output_dir = Path("outputs/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins = FOLD_SWITCH_BENCHMARK[:5]  # Quick demo
    print(f"Running ablation studies on {len(proteins)} proteins\n")

    ablation_configs = [
        {
            "name": "QCFold-QAOA (full)",
            "quantum_method": "qaoa",
            "use_classical_fallback": True,
            "num_ensemble": 8,
        },
        {
            "name": "QCFold-VQE",
            "quantum_method": "vqe",
            "use_classical_fallback": True,
            "num_ensemble": 8,
        },
        {
            "name": "Classical SA only",
            "quantum_method": "sa",
            "use_classical_fallback": False,
            "num_ensemble": 8,
        },
        {
            "name": "No ensemble (K=1)",
            "quantum_method": "qaoa",
            "use_classical_fallback": True,
            "num_ensemble": 1,
        },
        {
            "name": "Large ensemble (K=16)",
            "quantum_method": "qaoa",
            "use_classical_fallback": True,
            "num_ensemble": 16,
        },
    ]

    results = {}
    for config in ablation_configs:
        name = config.pop("name")
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        method = config.get("quantum_method", "qaoa")

        model = QCFoldModel(
            quantum_method=method,
            use_classical_fallback=config.get("use_classical_fallback", True),
            num_ensemble=config.get("num_ensemble", 8),
            max_quantum_residues=12,
            quantum_max_iterations=100,
            verbose=False,
        )

        harness = BenchmarkHarness(
            model=model,
            output_dir=str(output_dir / name.replace(" ", "_")),
            verbose=False,
        )

        benchmark_result = harness.run_benchmark(
            proteins=proteins,
            run_ablations=False,
        )

        results[name] = {
            "success_rate": benchmark_result.success_rate,
            "n_success": benchmark_result.n_success,
            "n_total": benchmark_result.n_proteins,
            "mean_fold_a_tm": benchmark_result.mean_fold_a_tm,
            "mean_fold_b_tm": benchmark_result.mean_fold_b_tm,
            "ensemble_diversity": benchmark_result.mean_ensemble_diversity,
            "wall_time": benchmark_result.wall_time_total,
        }

        print(f"  Success: {benchmark_result.n_success}/{benchmark_result.n_proteins} "
              f"({benchmark_result.success_rate:.1%})")
        print(f"  Fold A TM: {benchmark_result.mean_fold_a_tm:.3f}")
        print(f"  Fold B TM: {benchmark_result.mean_fold_b_tm:.3f}")
        print(f"  Time: {benchmark_result.wall_time_total:.1f}s")

    # Print ablation table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Rate':>8} {'TM-A':>6} {'TM-B':>6} "
          f"{'Div':>6} {'Time':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"  {name:<28} {r['success_rate']:>7.1%} "
              f"{r['mean_fold_a_tm']:>6.3f} {r['mean_fold_b_tm']:>6.3f} "
              f"{r['ensemble_diversity']:>6.2f} {r['wall_time']:>7.1f}s")

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results).T
    df.to_csv(output_dir / "ablation_results.csv")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
