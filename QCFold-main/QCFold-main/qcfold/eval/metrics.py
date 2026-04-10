"""
Evaluation metrics for fold-switching protein structure prediction.

Implements:
  1. TM-score (Template Modeling score) — primary metric
  2. RMSD after Kabsch superimposition
  3. GDT-TS (Global Distance Test - Total Score)
  4. lDDT (local Distance Difference Test) — approximate
  5. Ensemble metrics: MAT-P (precision) and MAT-R (recall)
  6. Fold-switching success rate
  7. Calibration metrics
  8. Per-residue distance deviation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StructureMetrics:
    """Metrics for a single structure prediction vs reference."""
    tm_score: float
    rmsd: float
    gdt_ts: float
    lddt: float
    num_aligned: int
    per_residue_dist: np.ndarray


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble prediction of fold-switching proteins."""
    # Per-conformation metrics
    fold_a_best_tm: float      # Best TM-score to Fold A across ensemble
    fold_b_best_tm: float      # Best TM-score to Fold B across ensemble
    both_predicted: bool       # Both conformations above threshold?
    success: bool              # Same as both_predicted

    # Ensemble quality
    mat_precision: float       # Average precision across generated structures
    mat_recall: float          # Coverage of known conformations
    ensemble_diversity: float  # Average pairwise TM-score within ensemble
    num_distinct_clusters: int # Number of structurally distinct clusters

    # Calibration
    confidence_fold_a: float   # Average confidence of Fold A predictions
    confidence_fold_b: float   # Average confidence of Fold B predictions

    # Efficiency
    num_structures: int        # Total structures generated
    structures_per_success: float  # Efficiency metric


def compute_tm_score(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    ref_length: Optional[int] = None,
) -> float:
    """Compute TM-score between predicted and reference CA coordinates.
    
    TM-score is normalized by reference length and is in [0, 1].
    TM-score > 0.5 indicates same general topology.
    TM-score > 0.6 is the threshold used for fold-switching success.
    
    Implementation follows Zhang & Skolnick (2004) Proteins 57:702-710.
    
    Args:
        pred_coords: (N, 3) predicted CA coordinates
        ref_coords: (M, 3) reference CA coordinates
        ref_length: normalization length (default: len(ref_coords))
    """
    if ref_length is None:
        ref_length = len(ref_coords)

    # Handle length mismatch by using the shorter length
    n = min(len(pred_coords), len(ref_coords))
    pred = pred_coords[:n]
    ref = ref_coords[:n]

    if n < 3:
        return 0.0

    # Superimpose using Kabsch
    pred_aligned = _kabsch_superimpose(pred, ref)

    # TM-score calculation
    d0 = 1.24 * (ref_length - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    distances = np.sqrt(np.sum((pred_aligned - ref) ** 2, axis=1))
    tm = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / ref_length

    return float(min(tm, 1.0))


def compute_rmsd(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> float:
    """Compute RMSD after Kabsch superimposition."""
    n = min(len(pred_coords), len(ref_coords))
    pred = pred_coords[:n]
    ref = ref_coords[:n]

    if n < 3:
        return float("inf")

    pred_aligned = _kabsch_superimpose(pred, ref)
    diff = pred_aligned - ref
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_gdt_ts(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute GDT-TS score."""
    n = min(len(pred_coords), len(ref_coords))
    pred = pred_coords[:n]
    ref = ref_coords[:n]

    pred_aligned = _kabsch_superimpose(pred, ref)
    distances = np.sqrt(np.sum((pred_aligned - ref) ** 2, axis=1))

    scores = [np.mean(distances < t) for t in thresholds]
    return float(np.mean(scores))


def compute_lddt(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    cutoff: float = 15.0,
) -> float:
    """Compute approximate lDDT (local Distance Difference Test).
    
    lDDT measures local structural accuracy by comparing distance
    deviations between all pairs within a cutoff distance.
    """
    n = min(len(pred_coords), len(ref_coords))
    pred = pred_coords[:n]
    ref = ref_coords[:n]

    # Compute distance matrices
    pred_dists = np.sqrt(np.sum(
        (pred[:, None] - pred[None, :]) ** 2, axis=-1
    ))
    ref_dists = np.sqrt(np.sum(
        (ref[:, None] - ref[None, :]) ** 2, axis=-1
    ))

    # Consider only pairs within cutoff in reference
    mask = (ref_dists < cutoff) & (np.eye(n) == 0)

    if mask.sum() == 0:
        return 0.0

    dist_diff = np.abs(pred_dists - ref_dists)

    scores = []
    for t in thresholds:
        preserved = (dist_diff[mask] < t).mean()
        scores.append(preserved)

    return float(np.mean(scores))


def evaluate_fold_switching(
    predictions: List[np.ndarray],      # List of predicted CA coords
    fold_a_coords: np.ndarray,          # (L, 3) Fold A reference
    fold_b_coords: np.ndarray,          # (L, 3) Fold B reference
    switch_region: Optional[Tuple[int, int]] = None,
    tm_threshold: float = 0.6,
    confidences: Optional[List[float]] = None,
) -> EnsembleMetrics:
    """Evaluate ensemble predictions against both known conformations.
    
    This is the primary evaluation function for fold-switching prediction.
    Success = at least one prediction has TM > threshold to each conformation.
    
    Args:
        predictions: list of (L, 3) predicted coordinate arrays
        fold_a_coords: reference Fold A coordinates
        fold_b_coords: reference Fold B coordinates
        switch_region: (start, end) tuple for fold-switching region evaluation
        tm_threshold: TM-score threshold for success (default 0.6)
        confidences: optional per-prediction confidence scores
    """
    num_preds = len(predictions)
    if num_preds == 0:
        return EnsembleMetrics(
            fold_a_best_tm=0.0, fold_b_best_tm=0.0,
            both_predicted=False, success=False,
            mat_precision=0.0, mat_recall=0.0,
            ensemble_diversity=0.0, num_distinct_clusters=0,
            confidence_fold_a=0.0, confidence_fold_b=0.0,
            num_structures=0, structures_per_success=float("inf"),
        )

    # Use switch region if specified
    ref_a = fold_a_coords
    ref_b = fold_b_coords
    preds = predictions

    if switch_region is not None:
        start, end = switch_region
        # Assume coordinates are indexed 0-based
        region_len = end - start
        ref_a = fold_a_coords[start:end]
        ref_b = fold_b_coords[start:end]
        preds = [p[start:end] for p in predictions]

    # Compute TM-scores to both conformations
    tm_scores_a = []
    tm_scores_b = []
    for pred in preds:
        tm_a = compute_tm_score(pred, ref_a)
        tm_b = compute_tm_score(pred, ref_b)
        tm_scores_a.append(tm_a)
        tm_scores_b.append(tm_b)

    tm_scores_a = np.array(tm_scores_a)
    tm_scores_b = np.array(tm_scores_b)

    fold_a_best_tm = float(np.max(tm_scores_a))
    fold_b_best_tm = float(np.max(tm_scores_b))
    both_predicted = (fold_a_best_tm >= tm_threshold and
                      fold_b_best_tm >= tm_threshold)

    # MAT-P (precision): average best-match TM for each prediction
    mat_precision = float(np.mean(np.maximum(tm_scores_a, tm_scores_b)))

    # MAT-R (recall): average best-match TM for each reference
    best_to_a = float(np.max(tm_scores_a))
    best_to_b = float(np.max(tm_scores_b))
    mat_recall = (best_to_a + best_to_b) / 2

    # Ensemble diversity: pairwise TM-scores within ensemble
    pairwise_tms = []
    for i in range(num_preds):
        for j in range(i + 1, num_preds):
            tm_ij = compute_tm_score(preds[i], preds[j])
            pairwise_tms.append(tm_ij)
    ensemble_diversity = float(np.mean(pairwise_tms)) if pairwise_tms else 0.0

    # Cluster count (simple: count groups with pairwise TM < 0.5)
    from scipy.cluster.hierarchy import fcluster, linkage
    if num_preds >= 2 and pairwise_tms:
        dist_matrix = np.zeros((num_preds, num_preds))
        idx = 0
        for i in range(num_preds):
            for j in range(i + 1, num_preds):
                dist_matrix[i, j] = 1.0 - pairwise_tms[idx]
                dist_matrix[j, i] = dist_matrix[i, j]
                idx += 1
        condensed = dist_matrix[np.triu_indices(num_preds, k=1)]
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=0.4, criterion="distance")
        num_clusters = len(set(clusters))
    else:
        num_clusters = 1

    # Confidence metrics
    conf_a = 0.0
    conf_b = 0.0
    if confidences is not None:
        best_a_idx = int(np.argmax(tm_scores_a))
        best_b_idx = int(np.argmax(tm_scores_b))
        conf_a = confidences[best_a_idx]
        conf_b = confidences[best_b_idx]

    # Efficiency
    structures_per_success = (
        num_preds if both_predicted else float("inf")
    )

    return EnsembleMetrics(
        fold_a_best_tm=fold_a_best_tm,
        fold_b_best_tm=fold_b_best_tm,
        both_predicted=both_predicted,
        success=both_predicted,
        mat_precision=mat_precision,
        mat_recall=mat_recall,
        ensemble_diversity=ensemble_diversity,
        num_distinct_clusters=num_clusters,
        confidence_fold_a=conf_a,
        confidence_fold_b=conf_b,
        num_structures=num_preds,
        structures_per_success=structures_per_success,
    )


def _kabsch_superimpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kabsch algorithm for optimal superimposition."""
    mob_center = mobile.mean(axis=0)
    tgt_center = target.mean(axis=0)
    mob_c = mobile - mob_center
    tgt_c = target - tgt_center

    H = mob_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign @ U.T

    return (R @ mob_c.T).T + tgt_center
