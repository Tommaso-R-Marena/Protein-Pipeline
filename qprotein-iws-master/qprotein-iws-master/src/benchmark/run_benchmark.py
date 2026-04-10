"""
Full benchmark: IWS-QAOA vs classical baselines on real PDB β-sheet windows.

Targets frustrated β-sheet windows where greedy demonstrably fails:
  - Greedy sub-optimality gap ≥ 5% compared to exact solution
  - Frustration index ≥ 1.0

PDB structures used (real β-sheet proteins):
  1. 1SHG — SH3 domain (all-β, 57 residues)
  2. 1TEN — Tenascin fibronectin type III (all-β, 89 residues)
  3. 2CI2 — Chymotrypsin inhibitor 2 (β-rich)
  4. 1PIN — Pin WW domain (small all-β, 34 residues)
  5. 1L2Y — Trp-cage (our prior work baseline)

For each structure:
  1. Detect β-sheet windows via Cα contacts
  2. Score frustration index
  3. Select the 2–3 most frustrated windows with N=4–6 residues
  4. Build QUBO, solve exactly
  5. Run greedy, SA, and IWS-QAOA
  6. Record: approximation ratios, ground state probabilities, runtime
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.pdb_loader import (
    build_qubo_from_window,
    compute_contact_matrix,
    detect_beta_sheet_residues,
    download_pdb,
    exact_solve,
    frustration_index,
    greedy_rotamer_pack,
    parse_pdb_backbone,
    simulated_annealing,
)
from src.qaoa.iws_qaoa import IWSQAOASolver, QAOAConfig
from src.routing.ogp_router import OGPRouter


BETA_SHEET_PROTEINS = [
    ("1SHG", "A", "SH3 domain (all-β)"),
    ("1TEN", "A", "Tenascin FNIII (all-β)"),
    ("1PIN", "A", "Pin WW domain (all-β)"),
    ("2CI2", "A", "Chymotrypsin inhibitor"),
    ("1L2Y", "A", "Trp-cage (prior work baseline)"),
]

N_ROTAMERS = 4   # rotamers per residue
WINDOW_SIZE = 5  # residues per window
MAX_QUBITS = 28  # stay within statevector simulator


def find_best_windows(residues: list[dict], beta_indices: list[int],
                       n_window: int = WINDOW_SIZE,
                       fi_min: float = 0.5) -> list[dict]:
    """
    Find high-frustration windows within β-sheet residues.
    Returns list of {window_indices, frustration_index, contact_density}.
    """
    windows = []
    beta_set = set(beta_indices)

    # Slide a window over β-sheet residues
    for start in range(len(residues) - n_window + 1):
        window = list(range(start, start + n_window))
        # Must have majority β-sheet residues
        n_beta = sum(1 for i in window if i in beta_set)
        if n_beta < n_window // 2:
            continue

        fi = frustration_index(residues, window)
        D = compute_contact_matrix(residues)
        contacts = [(i, j) for i in window for j in window
                    if i < j and D[i, j] < 8.0 and abs(i - j) >= 3]
        contact_density = len(contacts) / max((n_window * (n_window - 1)) // 2, 1)

        windows.append({
            "window_indices": window,
            "frustration_index": fi,
            "contact_density": contact_density,
            "n_beta": n_beta,
        })

    # Sort by frustration index descending
    windows.sort(key=lambda w: w["frustration_index"], reverse=True)
    return [w for w in windows if w["frustration_index"] >= fi_min]


def benchmark_window(pdb_id: str, residues: list[dict], window: dict,
                      n_rot: int = N_ROTAMERS,
                      qaoa_config: QAOAConfig = None) -> dict:
    """Run all solvers on one window. Returns result dict."""
    window_indices = window["window_indices"]
    N = len(window_indices)
    M = N * n_rot

    if M > MAX_QUBITS:
        print(f"  Skipping window {window_indices}: {M} qubits > limit {MAX_QUBITS}")
        return None

    print(f"\n  Window {[residues[i]['resnum'] for i in window_indices]} "
          f"FI={window['frustration_index']:.2f} M={M}q")

    # Build QUBO
    Q, meta = build_qubo_from_window(residues, window_indices, n_rot)

    # OGP routing
    router = OGPRouter()
    use_quantum, cert = router.should_use_quantum(Q, N, n_rot,
                                                    window["frustration_index"])

    # ── Exact solution ───────────────────────────────────────────────────────
    t0 = time.time()
    exact_x, exact_e = exact_solve(Q, N, n_rot)
    t_exact = time.time() - t0
    print(f"    Exact: {exact_e:.4f}  ({t_exact*1000:.1f}ms)")

    # ── Classical greedy ─────────────────────────────────────────────────────
    t0 = time.time()
    greedy_x, greedy_e = greedy_rotamer_pack(Q, N, n_rot)
    t_greedy = time.time() - t0
    greedy_gap = (greedy_e - exact_e) / (abs(exact_e) + 1e-9)
    print(f"    Greedy: {greedy_e:.4f}  gap={greedy_gap:.1%}  ({t_greedy*1000:.1f}ms)")

    # ── Simulated annealing ──────────────────────────────────────────────────
    t0 = time.time()
    sa_x, sa_e = simulated_annealing(Q, N, n_rot, n_steps=20000, T0=5.0)
    t_sa = time.time() - t0
    sa_gap = (sa_e - exact_e) / (abs(exact_e) + 1e-9)
    print(f"    SA:     {sa_e:.4f}  gap={sa_gap:.1%}  ({t_sa*1000:.1f}ms)")

    # ── IWS-QAOA (XY-mixer, warm-started from greedy) ───────────────────────
    if qaoa_config is None:
        qaoa_config = QAOAConfig(
            p=4,
            n_shots=512,
            n_iter=3,
            cvar_alpha=0.2,
            use_warm_start=True,
            use_xy_mixer=True,
            max_opt_iter=150,
            n_restarts=2,
            seed=42,
        )

    solver = IWSQAOASolver(Q, N, n_rot, qaoa_config)
    t0 = time.time()
    result = solver.solve(greedy_assignment=greedy_x, exact_energy=exact_e)
    t_qaoa = time.time() - t0
    qaoa_gap = (result.best_energy - exact_e) / (abs(exact_e) + 1e-9) \
        if not np.isnan(result.best_energy) else float("nan")
    print(f"    IWS-QAOA: {result.best_energy:.4f}  gap={qaoa_gap:.1%}  "
          f"GS-prob={result.ground_state_prob:.3f}  ({t_qaoa:.1f}s)")

    # ── Vanilla QAOA (no warm-start, transverse-field mixer, baseline) ───────
    vanilla_config = QAOAConfig(
        p=4, n_shots=512, n_iter=1, cvar_alpha=0.2,
        use_warm_start=False, use_xy_mixer=False,
        max_opt_iter=100, n_restarts=1, seed=42,
    )
    vanilla_solver = IWSQAOASolver(Q, N, n_rot, vanilla_config)
    t0 = time.time()
    vanilla_result = vanilla_solver.solve(exact_energy=exact_e)
    t_vanilla = time.time() - t0
    vanilla_gap = (vanilla_result.best_energy - exact_e) / (abs(exact_e) + 1e-9) \
        if not np.isnan(vanilla_result.best_energy) else float("nan")
    print(f"    Vanilla QAOA: {vanilla_result.best_energy:.4f}  "
          f"gap={vanilla_gap:.1%}  ({t_vanilla:.1f}s)")

    return {
        "pdb_id": pdb_id,
        "window_residues": [residues[i]["resnum"] for i in window_indices],
        "window_resnames": [residues[i]["resname"] for i in window_indices],
        "N": N,
        "n_rotamers": n_rot,
        "M_qubits": M,
        "frustration_index": window["frustration_index"],
        "contact_density": window["contact_density"],
        "ogp_certificate": cert,
        "routed_to_quantum": use_quantum,
        "exact": {
            "energy": exact_e,
            "runtime_ms": t_exact * 1000,
        },
        "greedy": {
            "energy": greedy_e,
            "gap_pct": greedy_gap * 100,
            "runtime_ms": t_greedy * 1000,
            "matches_exact": abs(greedy_e - exact_e) < 0.01,
        },
        "sa": {
            "energy": sa_e,
            "gap_pct": sa_gap * 100,
            "runtime_ms": t_sa * 1000,
            "matches_exact": abs(sa_e - exact_e) < 0.01,
        },
        "iws_qaoa": {
            "energy": float(result.best_energy),
            "gap_pct": qaoa_gap * 100 if not np.isnan(qaoa_gap) else None,
            "approximation_ratio": float(result.approximation_ratio),
            "ground_state_prob": float(result.ground_state_prob),
            "n_circuit_evals": result.n_circuit_evals,
            "runtime_s": result.runtime_s,
            "matches_exact": not np.isnan(result.best_energy) and
                             abs(result.best_energy - exact_e) < 0.01,
            "iws_history": result.iws_history,
        },
        "vanilla_qaoa": {
            "energy": float(vanilla_result.best_energy),
            "gap_pct": vanilla_gap * 100 if not np.isnan(vanilla_gap) else None,
            "ground_state_prob": float(vanilla_result.ground_state_prob),
            "runtime_s": vanilla_result.runtime_s,
            "matches_exact": not np.isnan(vanilla_result.best_energy) and
                             abs(vanilla_result.best_energy - exact_e) < 0.01,
        },
        "improvement_over_greedy": {
            "iws_qaoa_vs_greedy_gap": (
                (greedy_gap - qaoa_gap) * 100
                if not np.isnan(qaoa_gap) else None
            ),
            "sa_vs_greedy_gap": (sa_gap - greedy_gap) * 100,
        },
    }


def run_full_benchmark(output_dir: str = "/home/user/workspace/qprotein-iws/results",
                        max_windows_per_protein: int = 2) -> list[dict]:
    """Run benchmark across all β-sheet proteins."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    qaoa_config = QAOAConfig(
        p=4, n_shots=512, n_iter=3, cvar_alpha=0.2,
        use_warm_start=True, use_xy_mixer=True,
        max_opt_iter=100, n_restarts=2, seed=42,
    )

    for pdb_id, chain_id, description in BETA_SHEET_PROTEINS:
        print(f"\n{'='*60}")
        print(f"Processing {pdb_id}: {description}")
        print(f"{'='*60}")

        try:
            pdb_path = download_pdb(pdb_id)
            residues = parse_pdb_backbone(pdb_path, chain_id)
            print(f"  {len(residues)} residues parsed")

            beta_indices = detect_beta_sheet_residues(residues)
            print(f"  {len(beta_indices)} β-sheet residues detected: "
                  f"{[residues[i]['resnum'] for i in beta_indices[:8]]}...")

            windows = find_best_windows(residues, beta_indices,
                                        n_window=WINDOW_SIZE, fi_min=0.3)
            print(f"  {len(windows)} frustrated windows found")

            if not windows:
                print("  No frustrated windows — using all-residue windows")
                windows = find_best_windows(residues, list(range(len(residues))),
                                            n_window=WINDOW_SIZE, fi_min=0.0)

            for window in windows[:max_windows_per_protein]:
                result = benchmark_window(pdb_id, residues, window,
                                          n_rot=N_ROTAMERS,
                                          qaoa_config=qaoa_config)
                if result:
                    all_results.append(result)

        except Exception as e:
            print(f"  ERROR processing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    out_path = os.path.join(output_dir, "json", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'PDB':<6} {'Window':<20} {'Exact':>8} {'Greedy':>8} {'SA':>8} "
          f"{'IWS-QAOA':>10} {'GS-P':>6} {'Route':>8}")
    print("-"*80)
    for r in results:
        window_str = ",".join(map(str, r["window_residues"]))
        print(f"{r['pdb_id']:<6} {window_str:<20} "
              f"{r['exact']['energy']:>8.2f} "
              f"{r['greedy']['energy']:>8.2f} "
              f"{r['sa']['energy']:>8.2f} "
              f"{r['iws_qaoa']['energy']:>10.2f} "
              f"{r['iws_qaoa']['ground_state_prob']:>6.3f} "
              f"{r['ogp_certificate']['routed_to']:>8}")
