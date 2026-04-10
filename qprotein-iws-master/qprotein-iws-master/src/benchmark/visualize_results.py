"""
Publication-quality figures for IWS-QAOA benchmark results.

Produces:
  Fig 1: Bar chart — approximation ratios across solvers and windows
  Fig 2: Ground state probability vs frustration index (scatter)
  Fig 3: IWS convergence curves (energy vs round)
  Fig 4: OGP certificate distributions (routing decisions)
  Fig 5: Quantum vs classical gap improvement for frustrated windows
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = "/home/user/workspace/qprotein-iws/results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "greedy": "#E74C3C",
    "sa": "#F39C12",
    "vanilla_qaoa": "#3498DB",
    "iws_qaoa": "#27AE60",
    "exact": "#2C3E50",
}


def load_results(path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "json", "benchmark_results.json")
    with open(path) as f:
        return json.load(f)


def fig1_approximation_ratios(results: list[dict]):
    """Bar chart of approximation ratios (energy / exact_energy) per window."""
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = []
    greedy_gaps = []
    sa_gaps = []
    vanilla_gaps = []
    iws_gaps = []
    frustrated = []

    for r in results:
        label = f"{r['pdb_id']}\n{','.join(str(x) for x in r['window_residues'][:3])}..."
        labels.append(label)
        exact_e = r["exact"]["energy"]
        
        def safe_gap(e):
            if e is None or e != e:  # nan check
                return 0.0
            if abs(exact_e) < 1e-9:
                return 0.0
            return max(0.0, (e - exact_e) / abs(exact_e)) * 100

        greedy_gaps.append(safe_gap(r["greedy"]["energy"]))
        sa_gaps.append(safe_gap(r["sa"]["energy"]))
        vanilla_gaps.append(safe_gap(r["vanilla_qaoa"]["energy"]))
        iws_gaps.append(safe_gap(r["iws_qaoa"]["energy"]))
        frustrated.append(r["frustration_index"] > 1.0)

    n = len(labels)
    x = np.arange(n)
    w = 0.18

    bars_g = ax.bar(x - 1.5*w, greedy_gaps, w, label="Greedy",
                    color=COLORS["greedy"], alpha=0.85)
    bars_s = ax.bar(x - 0.5*w, sa_gaps, w, label="SA",
                    color=COLORS["sa"], alpha=0.85)
    bars_v = ax.bar(x + 0.5*w, vanilla_gaps, w, label="Vanilla QAOA",
                    color=COLORS["vanilla_qaoa"], alpha=0.85)
    bars_q = ax.bar(x + 1.5*w, iws_gaps, w, label="IWS-QAOA (ours)",
                    color=COLORS["iws_qaoa"], alpha=0.85)

    # Highlight frustrated windows
    for i, frust in enumerate(frustrated):
        if frust:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="purple",
                       label="_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Suboptimality gap (%)\n(0% = optimal)")
    ax.set_title("Approximation Quality: IWS-QAOA vs Classical Baselines\n"
                 "(shaded = frustrated β-sheet window, FI > 1.0)")
    ax.legend(loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(bottom=-1)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_approximation_ratios.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig2_ground_state_vs_frustration(results: list[dict]):
    """Scatter: ground state probability vs frustration index, colored by routing."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        fi = r["frustration_index"]
        gsp = r["iws_qaoa"]["ground_state_prob"]
        routed = r["ogp_certificate"]["routed_to"]
        color = COLORS["iws_qaoa"] if routed == "quantum" else COLORS["greedy"]
        marker = "o" if routed == "quantum" else "s"
        ax.scatter(fi, gsp, color=color, marker=marker, s=120, alpha=0.85,
                   edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(r["pdb_id"], (fi, gsp),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Threshold line
    ax.axvline(1.0, color="purple", linestyle="--", linewidth=1.2, alpha=0.6,
               label="OGP threshold (FI=1.0)")

    q_patch = mpatches.Patch(color=COLORS["iws_qaoa"], label="Routed → Quantum")
    c_patch = mpatches.Patch(color=COLORS["greedy"], label="Routed → Classical")
    ax.legend(handles=[q_patch, c_patch,
                        mpatches.Patch(color="purple", alpha=0.6,
                                        label="OGP threshold")],
              loc="lower right")

    ax.set_xlabel("Local Frustration Index (FI)")
    ax.set_ylabel("Ground State Probability P(optimal)")
    ax.set_title("IWS-QAOA: Ground State Probability vs. Structural Frustration")
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_gs_vs_frustration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig3_iws_convergence(results: list[dict]):
    """IWS convergence: energy vs iteration for each window."""
    # Find results with multi-round history
    multi_round = [r for r in results
                   if len(r["iws_qaoa"]["iws_history"]) > 1]
    if not multi_round:
        multi_round = results

    n_plots = min(4, len(multi_round))
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, r in zip(axes, multi_round[:n_plots]):
        history = r["iws_qaoa"]["iws_history"]
        rounds = [h["round"] + 1 for h in history]
        energies = [h["best_energy"] if h["best_energy"] != float("inf")
                    else None for h in history]

        # Filter None
        valid = [(rnd, e) for rnd, e in zip(rounds, energies) if e is not None]
        if valid:
            r_valid, e_valid = zip(*valid)
            ax.plot(r_valid, e_valid, "o-", color=COLORS["iws_qaoa"],
                    linewidth=2, markersize=7, label="IWS-QAOA")

        # Horizontal lines for baselines
        exact_e = r["exact"]["energy"]
        greedy_e = r["greedy"]["energy"]
        sa_e = r["sa"]["energy"]

        ax.axhline(exact_e, color=COLORS["exact"], linestyle="-",
                   linewidth=1.5, alpha=0.9, label=f"Exact ({exact_e:.1f})")
        ax.axhline(greedy_e, color=COLORS["greedy"], linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"Greedy ({greedy_e:.1f})")
        ax.axhline(sa_e, color=COLORS["sa"], linestyle=":",
                   linewidth=1.2, alpha=0.7, label=f"SA ({sa_e:.1f})")

        ax.set_xlabel("IWS Round")
        ax.set_ylabel("Best Energy Found")
        title = f"{r['pdb_id']} res {r['window_residues'][0]}–{r['window_residues'][-1]}"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xticks(rounds)

    fig.suptitle("IWS-QAOA Convergence: Energy Improvement per Iteration",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_iws_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig4_ogp_routing(results: list[dict]):
    """OGP certificate scatter: ρ vs FI, colored by routing."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for r in results:
        cert = r["ogp_certificate"]
        rho = cert["rho"]
        fi = cert["frustration_index"]
        routed = cert["routed_to"]
        color = COLORS["iws_qaoa"] if routed == "quantum" else COLORS["greedy"]
        ax.scatter(rho, fi, color=color, s=140, alpha=0.85,
                   edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(r["pdb_id"], (rho, fi),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Threshold lines
    ax.axvline(0.30, color="gray", linestyle="--", linewidth=1.2,
               label="ρ threshold (0.30)")
    ax.axhline(1.0, color="purple", linestyle="--", linewidth=1.2,
               label="FI threshold (1.0)")

    # Quadrant labels
    ax.text(0.05, 1.5, "Low ρ\nHigh FI\n(Classical)", fontsize=8,
            color="gray", alpha=0.7)
    ax.text(0.35, 1.5, "High ρ\nHigh FI\n→ Quantum", fontsize=8,
            color=COLORS["iws_qaoa"], alpha=0.9)
    ax.text(0.35, 0.2, "High ρ\nLow FI\n(Borderline)", fontsize=8,
            color="gray", alpha=0.7)

    q_patch = mpatches.Patch(color=COLORS["iws_qaoa"], label="Routed → Quantum")
    c_patch = mpatches.Patch(color=COLORS["greedy"], label="Routed → Classical")
    ax.legend(handles=[q_patch, c_patch], loc="upper left", fontsize=9)

    ax.set_xlabel("Interaction Density ρ")
    ax.set_ylabel("Frustration Index (FI)")
    ax.set_title("OGP Routing Certificate: ρ vs. FI")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_ogp_routing.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig5_quantum_vs_classical_gap(results: list[dict]):
    """
    Key result figure: IWS-QAOA gap improvement over greedy for frustrated windows.
    Shows where quantum helps most.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = []
    greedy_gaps = []
    iws_gaps = []
    sa_gaps = []
    colors_list = []

    for r in results:
        exact_e = r["exact"]["energy"]
        def safe_gap(e):
            if e is None or (isinstance(e, float) and e != e):
                return None
            if abs(exact_e) < 1e-9:
                return 0.0
            return max(0.0, (e - exact_e) / abs(exact_e)) * 100

        gg = safe_gap(r["greedy"]["energy"])
        qg = safe_gap(r["iws_qaoa"]["energy"])
        sg = safe_gap(r["sa"]["energy"])
        fi = r["frustration_index"]

        if gg is None or qg is None:
            continue

        label = f"{r['pdb_id']}\nFI={fi:.1f}"
        labels.append(label)
        greedy_gaps.append(gg)
        iws_gaps.append(qg)
        sa_gaps.append(sg if sg is not None else 0.0)
        # Color by frustration
        colors_list.append("#27AE60" if fi > 1.0 else "#3498DB")

    n = len(labels)
    if n == 0:
        print("No valid data for Fig 5")
        plt.close(fig)
        return None

    x = np.arange(n)
    w = 0.25

    ax.bar(x - w, greedy_gaps, w, color=COLORS["greedy"], alpha=0.85,
           label="Greedy", zorder=2)
    ax.bar(x, sa_gaps, w, color=COLORS["sa"], alpha=0.85, label="SA", zorder=2)
    ax.bar(x + w, iws_gaps, w, color=COLORS["iws_qaoa"], alpha=0.85,
           label="IWS-QAOA (ours)", zorder=2)

    # Arrows showing improvement
    for i in range(n):
        if greedy_gaps[i] > 0.01 and iws_gaps[i] < greedy_gaps[i] - 0.01:
            ax.annotate("", xy=(x[i] + w, iws_gaps[i]),
                        xytext=(x[i] - w, greedy_gaps[i]),
                        arrowprops=dict(arrowstyle="->", color="purple",
                                        lw=1.5, alpha=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Suboptimality gap (%)\n(0% = global optimum)")
    ax.set_title("IWS-QAOA vs Classical Baselines: Gap from Optimal\n"
                 "(Green = frustrated window FI>1 | Purple arrows = quantum improvement)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(bottom=-1)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_quantum_vs_classical.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def make_summary_table(results: list[dict]) -> str:
    """Generate markdown summary table."""
    rows = []
    header = ("| PDB | Window | FI | Qubits | Exact | Greedy gap | SA gap | "
               "IWS-QAOA gap | GS-Prob | Route |\n"
               "|-----|--------|----|--------|-------|-----------|--------|"
               "-------------|---------|-------|\n")
    for r in results:
        exact_e = r["exact"]["energy"]
        def safe_gap(e):
            if e is None or (isinstance(e, float) and e != e):
                return "—"
            if abs(exact_e) < 1e-9:
                return "0.0%"
            gap = max(0.0, (e - exact_e) / abs(exact_e)) * 100
            return f"{gap:.1f}%"

        win = f"{r['window_residues'][0]}–{r['window_residues'][-1]}"
        fi = r["frustration_index"]
        M = r["M_qubits"]
        exact = f"{exact_e:.2f}"
        gg = safe_gap(r["greedy"]["energy"])
        sg = safe_gap(r["sa"]["energy"])
        qg = safe_gap(r["iws_qaoa"]["energy"])
        gsp = f"{r['iws_qaoa']['ground_state_prob']:.3f}"
        route = r["ogp_certificate"]["routed_to"]
        rows.append(f"| {r['pdb_id']} | {win} | {fi:.2f} | {M} | {exact} | "
                    f"{gg} | {sg} | {qg} | {gsp} | {route} |")

    return header + "\n".join(rows)


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} benchmark results")

    fig1_approximation_ratios(results)
    fig2_ground_state_vs_frustration(results)
    fig3_iws_convergence(results)
    fig4_ogp_routing(results)
    fig5_quantum_vs_classical_gap(results)

    table = make_summary_table(results)
    table_path = os.path.join(RESULTS_DIR, "tables", "summary_table.md")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# IWS-QAOA Benchmark Results\n\n")
        f.write(table)
    print(f"Saved table to {table_path}")
    print("\n" + table)
