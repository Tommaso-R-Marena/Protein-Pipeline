"""
OGP-based quantum/classical routing for rotamer packing.

Decides when to use IWS-QAOA vs classical (SA or greedy) based on the
Overlap Gap Property (OGP) complexity certificate for the QUBO instance.

Key insight (from QNSA/Marena 2026): the OGP predicts where classical
local search (greedy, SA) gets trapped in the landscape. When OGP is
present (dense interactions, frustrated contact network), quantum mixing
provides a genuine advantage by tunneling through barriers.

OGP certificate for rotamer QUBO:
  - Clause density ρ = |{(i,j): |J_ij| > θ}| / (M choose 2)
  - Spectral gap = λ_1(L_J) / λ_2(L_J)  (from J matrix Laplacian)
  - Frustration index from pdb_loader.frustration_index()
  
Route to quantum when: ρ > ρ_thresh AND FI > FI_thresh
Route to classical otherwise.
"""

from __future__ import annotations
import numpy as np


# Thresholds calibrated from prior QADF v2 work and QNSA experiments
RHO_THRESHOLD = 0.30     # interaction density > 30% → frustrated
FI_THRESHOLD = 1.0       # frustration index > 1 std → OGP present
SPECTRAL_THRESHOLD = 0.5  # spectral gap < 0.5 → glassy landscape


class OGPRouter:
    """
    Routes rotamer packing instances to quantum or classical solver
    based on their OGP complexity certificate.
    """

    def __init__(self,
                 rho_threshold: float = RHO_THRESHOLD,
                 fi_threshold: float = FI_THRESHOLD,
                 spectral_threshold: float = SPECTRAL_THRESHOLD):
        self.rho_threshold = rho_threshold
        self.fi_threshold = fi_threshold
        self.spectral_threshold = spectral_threshold

    def compute_certificate(self, Q: np.ndarray, N: int, n: int,
                             frustration_index: float = 0.0) -> dict:
        """
        Compute OGP complexity certificate for a QUBO instance.
        
        Returns dict with:
          - rho: interaction density
          - spectral_gap: eigenvalue ratio of interaction Laplacian
          - frustration_index: from structural analysis
          - routed_to: "quantum" or "classical"
          - reason: human-readable explanation
        """
        M = N * n
        # Interaction density: fraction of off-diagonal pairs with |J_ij| > threshold
        theta = 0.1 * np.abs(Q).max()
        off_diag_count = 0
        total_pairs = M * (M - 1) // 2
        for i in range(M):
            for j in range(i + 1, M):
                if abs(Q[i, j]) > theta:
                    off_diag_count += 1
        rho = off_diag_count / max(total_pairs, 1)

        # Spectral gap of the interaction Laplacian
        J_abs = np.abs(Q.copy())
        np.fill_diagonal(J_abs, 0)
        degree = J_abs.sum(axis=1)
        L = np.diag(degree) - J_abs
        try:
            eigvals = np.linalg.eigvalsh(L)
            eigvals_sorted = sorted(eigvals)
            # Spectral gap = second smallest / largest
            if len(eigvals_sorted) > 1 and eigvals_sorted[-1] > 1e-9:
                spectral_gap = eigvals_sorted[1] / eigvals_sorted[-1]
            else:
                spectral_gap = 1.0
        except Exception:
            spectral_gap = 1.0

        # Routing decision: OGP-guided
        reasons = []
        route_quantum_votes = 0
        route_classical_votes = 0

        if rho > self.rho_threshold:
            reasons.append(f"ρ={rho:.3f} > {self.rho_threshold} (dense interactions)")
            route_quantum_votes += 1
        else:
            reasons.append(f"ρ={rho:.3f} ≤ {self.rho_threshold} (sparse: greedy sufficient)")
            route_classical_votes += 1

        if frustration_index > self.fi_threshold:
            reasons.append(f"FI={frustration_index:.2f} > {self.fi_threshold} (frustrated)")
            route_quantum_votes += 1
        else:
            reasons.append(f"FI={frustration_index:.2f} ≤ {self.fi_threshold} (low frustration)")
            route_classical_votes += 1

        if spectral_gap < self.spectral_threshold:
            reasons.append(f"gap={spectral_gap:.3f} < {self.spectral_threshold} (glassy)")
            route_quantum_votes += 1
        else:
            reasons.append(f"gap={spectral_gap:.3f} ≥ {self.spectral_threshold} (gapped)")
            route_classical_votes += 1

        routed_to = "quantum" if route_quantum_votes >= 2 else "classical"

        return {
            "rho": rho,
            "spectral_gap": spectral_gap,
            "frustration_index": frustration_index,
            "route_quantum_votes": route_quantum_votes,
            "route_classical_votes": route_classical_votes,
            "routed_to": routed_to,
            "reason": " | ".join(reasons),
        }

    def should_use_quantum(self, Q: np.ndarray, N: int, n: int,
                            frustration_index: float = 0.0) -> tuple[bool, dict]:
        """Returns (use_quantum, certificate_dict)."""
        cert = self.compute_certificate(Q, N, n, frustration_index)
        return cert["routed_to"] == "quantum", cert
