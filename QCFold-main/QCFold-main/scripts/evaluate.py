#!/usr/bin/env python3
"""
QCFold Benchmark Evaluation Script

Runs the full evaluation pipeline on the fold-switching protein benchmark.
Produces results tables, statistical analysis, and comparison vs baselines.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --quick  # Quick demo on 5 proteins
    python scripts/evaluate.py --ablations  # With ablation studies
"""

import sys
import os
import argparse
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcfold.config import QCFoldConfig
from qcfold.models.qcfold_model import QCFoldModel
from qcfold.eval.benchmark_harness import BenchmarkHarness, BASELINE_RESULTS
from qcfold.data.benchmark import (
    get_benchmark_proteins, FOLD_SWITCH_BENCHMARK,
    DIFFICULTY_TIERS,
)


def main():
    parser = argparse.ArgumentParser(description="QCFold Benchmark Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true",
                       help="Quick demo on subset of proteins")
    parser.add_argument("--ablations", action="store_true",
                       help="Run ablation studies")
    parser.add_argument("--quantum-method", type=str, default="qaoa",
                       choices=["qaoa", "vqe", "sa"],
                       help="Optimization method")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print("QCFold: Quantum-Classical Fold-Switching Protein Structure Prediction")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Quantum method: {args.quantum_method}")
    print(f"  Ablations: {args.ablations}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output: {args.output_dir}")

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config,
    )
    if os.path.exists(config_path):
        config = QCFoldConfig.from_yaml(config_path)
    else:
        config = QCFoldConfig()

    # Override with command-line args
    config.quantum.enabled = True
    config.output_dir = args.output_dir

    # Select proteins
    if args.quick:
        proteins = FOLD_SWITCH_BENCHMARK[:5]
        print(f"\nQuick mode: evaluating {len(proteins)} proteins")
    else:
        proteins = FOLD_SWITCH_BENCHMARK
        print(f"\nFull benchmark: evaluating {len(proteins)} proteins")

    # Initialize model
    model = QCFoldModel(
        encoder_type="onehot",
        num_candidates=config.generator.num_candidates,
        quantum_method=args.quantum_method,
        quantum_layers=config.quantum.num_layers,
        quantum_max_iterations=config.quantum.max_iterations,
        quantum_lr=config.quantum.lr,
        quantum_backend=config.quantum.backend,
        max_quantum_residues=config.quantum.max_region_size,
        use_classical_fallback=config.quantum.use_classical_fallback,
        num_ensemble=config.ensemble.num_conformations,
        diversity_weight=config.ensemble.diversity_weight,
        physics_weight=config.ranking.physics_weight,
        seed=config.seed,
        verbose=args.verbose,
    )

    # Run benchmark
    harness = BenchmarkHarness(
        model=model,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print(f"\n{'='*70}")
    print("Running benchmark evaluation...")
    print(f"{'='*70}")

    result = harness.run_benchmark(
        proteins=proteins,
        run_ablations=args.ablations,
        num_seeds=args.num_seeds,
    )

    # Print results summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\nQCFold Performance:")
    print(f"  Proteins evaluated: {result.n_proteins}")
    print(f"  Successes (both folds predicted): {result.n_success}")
    print(f"  Success rate: {result.success_rate:.1%}")
    print(f"  95% CI: [{result.success_rate_ci[0]:.1%}, {result.success_rate_ci[1]:.1%}]")
    print(f"  Mean Fold A TM-score: {result.mean_fold_a_tm:.4f}")
    print(f"  Mean Fold B TM-score: {result.mean_fold_b_tm:.4f}")
    print(f"  Mean ensemble diversity (RMSD): {result.mean_ensemble_diversity:.2f} Å")
    print(f"  Total wall time: {result.wall_time_total:.1f}s")

    # Comparison vs baselines
    print(f"\n{'='*70}")
    print("COMPARISON VS BASELINES (92-protein benchmark)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Success Rate':>12} {'Source':>40}")
    print("-" * 77)
    for name, comp in result.comparison_vs_baselines.items():
        print(f"  {name:<23} {comp['baseline_rate']:>10.1%}   {comp['source']}")
    print(f"  {'QCFold (ours)':<23} {result.success_rate:>10.1%}   This work")

    # Stratified results
    if result.stratified_results.get("by_difficulty"):
        print(f"\n{'='*70}")
        print("STRATIFIED BY DIFFICULTY")
        print(f"{'='*70}")
        for diff, stats in result.stratified_results["by_difficulty"].items():
            print(f"  {diff:<15} {stats['n_success']}/{stats['n_total']} "
                  f"({stats['success_rate']:.1%})")

    if result.stratified_results.get("by_training_status"):
        print(f"\nSTRATIFIED BY TRAINING STATUS")
        for status, stats in result.stratified_results["by_training_status"].items():
            print(f"  {status:<20} {stats['n_success']}/{stats['n_total']} "
                  f"({stats['success_rate']:.1%})")

    # Ablation results
    if result.ablation_summary:
        print(f"\n{'='*70}")
        print("ABLATION RESULTS")
        print(f"{'='*70}")
        for abl_name, abl_stats in result.ablation_summary.items():
            if "success_rate" in abl_stats:
                print(f"  {abl_name:<25} Success: {abl_stats['success_rate']:.1%}, "
                      f"TM-A: {abl_stats.get('mean_fold_a_tm', 0):.3f}, "
                      f"TM-B: {abl_stats.get('mean_fold_b_tm', 0):.3f}")
            elif "mean_energy" in abl_stats:
                print(f"  {abl_name:<25} Energy: {abl_stats['mean_energy']:.4f}")

    # Per-protein details
    print(f"\n{'='*70}")
    print("PER-PROTEIN RESULTS")
    print(f"{'='*70}")
    print(f"{'Protein':<25} {'TM-A':>6} {'TM-B':>6} {'Success':>8} {'Method':>8} {'Time':>6}")
    print("-" * 65)
    for r in result.per_protein_results:
        if "error" in r:
            print(f"  {r['protein_name']:<23} {'ERROR':>6}")
        else:
            success_str = "YES" if r["both_predicted"] else "no"
            print(f"  {r['protein_name']:<23} {r['fold_a_tm']:>6.3f} "
                  f"{r['fold_b_tm']:>6.3f} {success_str:>8} "
                  f"{r['method_used']:>8} {r['wall_time']:>5.1f}s")

    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  benchmark_results.json")
    print(f"  per_protein_results.csv")


if __name__ == "__main__":
    main()
