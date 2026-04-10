"""
Statistical analysis for benchmark evaluation.

Implements:
  1. Wilcoxon signed-rank test (paired comparison)
  2. Bootstrap confidence intervals
  3. Multiple-testing correction (Bonferroni, BH-FDR)
  4. Calibration analysis (reliability diagrams)
  5. Effect size computation (Cohen's d, rank-biserial)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_name: str
    test_name: str
    n_samples: int


def wilcoxon_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method_a: str = "method_A",
    method_b: str = "method_B",
    metric: str = "TM-score",
    alpha: float = 0.05,
) -> ComparisonResult:
    """Wilcoxon signed-rank test for paired comparison.
    
    Appropriate for non-normal paired observations (e.g., TM-scores
    on the same proteins from two different methods).
    """
    diffs = scores_a - scores_b
    non_zero_diffs = diffs[diffs != 0]

    if len(non_zero_diffs) < 5:
        # Too few non-zero differences
        return ComparisonResult(
            method_a=method_a, method_b=method_b, metric=metric,
            mean_a=float(np.mean(scores_a)),
            mean_b=float(np.mean(scores_b)),
            mean_diff=float(np.mean(diffs)),
            ci_lower=float(np.nan), ci_upper=float(np.nan),
            p_value=1.0, significant=False,
            effect_size=0.0, effect_size_name="rank_biserial",
            test_name="Wilcoxon (insufficient data)",
            n_samples=len(scores_a),
        )

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    n = len(non_zero_diffs)

    # Rank-biserial effect size
    r = 1 - (2 * stat) / (n * (n + 1))

    # Bootstrap CI for mean difference
    ci_lower, ci_upper = bootstrap_ci(diffs, alpha=alpha)

    return ComparisonResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        mean_a=float(np.mean(scores_a)),
        mean_b=float(np.mean(scores_b)),
        mean_diff=float(np.mean(diffs)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        significant=float(p_value) < alpha,
        effect_size=float(r),
        effect_size_name="rank_biserial",
        test_name="Wilcoxon signed-rank",
        n_samples=len(scores_a),
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.
    
    Returns:
        (lower, upper) bounds at (1-alpha)*100% confidence level
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    stat_fn = np.mean if statistic == "mean" else np.median
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def success_rate_ci(
    n_success: int,
    n_total: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Wilson score interval for success rate.
    
    Returns:
        (rate, lower, upper)
    """
    if n_total == 0:
        return 0.0, 0.0, 0.0

    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator

    return float(p), float(center - margin), float(center + margin)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size for two independent groups."""
    na, nb = len(group_a), len(group_b)
    pooled_std = np.sqrt(
        ((na - 1) * np.var(group_a, ddof=1) +
         (nb - 1) * np.var(group_b, ddof=1)) / (na + nb - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def calibration_analysis(
    predicted_confidence: np.ndarray,
    actual_success: np.ndarray,
    num_bins: int = 10,
) -> Dict:
    """Compute calibration metrics: ECE, MCE, reliability diagram data.
    
    Args:
        predicted_confidence: (N,) predicted confidence [0, 1]
        actual_success: (N,) binary actual outcome
    
    Returns:
        dict with ECE, MCE, and per-bin calibration data
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_data = []

    ece = 0.0
    mce = 0.0

    for b in range(num_bins):
        mask = ((predicted_confidence >= bin_edges[b]) &
                (predicted_confidence < bin_edges[b + 1]))
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = predicted_confidence[mask].mean()
        avg_acc = actual_success[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += gap * (n_bin / len(predicted_confidence))
        mce = max(mce, gap)
        bin_data.append({
            "bin_lower": float(bin_edges[b]),
            "bin_upper": float(bin_edges[b + 1]),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
            "count": int(n_bin),
            "gap": float(gap),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_data,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """Bonferroni multiple testing correction."""
    n = len(p_values)
    adjusted = [(min(p * n, 1.0), min(p * n, 1.0) < alpha) for p in p_values]
    return adjusted


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n)

    for rank, idx in enumerate(sorted_indices, 1):
        adjusted[idx] = p_values[idx] * n / rank

    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[sorted_indices[i]] = min(
            adjusted[sorted_indices[i]],
            adjusted[sorted_indices[i + 1]] if i + 1 < n else 1.0,
        )

    adjusted = np.minimum(adjusted, 1.0)
    return [(float(adjusted[i]), adjusted[i] < alpha) for i in range(n)]
