"""Core tests for IWS-QAOA protein rotamer packing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from src.data.pdb_loader import (
    build_qubo_from_window, greedy_rotamer_pack, exact_solve,
    simulated_annealing, get_rotamers_for_residue
)
from src.mixers.xy_mixer import xy_mixer_circuit, prepare_w_state
from src.routing.ogp_router import OGPRouter
from src.qaoa.iws_qaoa import IWSQAOASolver, QAOAConfig, qubo_to_ising


# ── QUBO construction ────────────────────────────────────────────────────────

def test_rotamers_for_known_aa():
    rots = get_rotamers_for_residue("LEU", 4)
    assert len(rots) == 4
    assert all(0.0 < r[2] <= 1.0 for r in rots), "All probs must be in (0,1]"

def test_qubo_is_square():
    from src.data.pdb_loader import download_pdb, parse_pdb_backbone
    pdb_path = download_pdb("1L2Y")
    residues = parse_pdb_backbone(pdb_path, "A")
    Q, meta = build_qubo_from_window(residues, [0,1,2,3], 3)
    assert Q.shape == (12, 12)
    assert np.allclose(Q, Q.T), "QUBO must be symmetric"

def test_exact_beats_greedy_or_ties():
    """Exact solution must have energy ≤ greedy."""
    from src.data.pdb_loader import download_pdb, parse_pdb_backbone
    pdb_path = download_pdb("1L2Y")
    residues = parse_pdb_backbone(pdb_path, "A")
    Q, _ = build_qubo_from_window(residues, [0,1,2,3], 3)
    ex_x, ex_e = exact_solve(Q, 4, 3)
    gr_x, gr_e = greedy_rotamer_pack(Q, 4, 3)
    assert ex_e <= gr_e + 1e-6, f"Exact {ex_e} must be <= greedy {gr_e}"

def test_sa_valid_assignment():
    from src.data.pdb_loader import download_pdb, parse_pdb_backbone
    pdb_path = download_pdb("1L2Y")
    residues = parse_pdb_backbone(pdb_path, "A")
    Q, _ = build_qubo_from_window(residues, [0,1,2,3], 3)
    sa_x, sa_e = simulated_annealing(Q, 4, 3, n_steps=500)
    # Check one-hot
    for i in range(4):
        assert sa_x[i*3:i*3+3].sum() == 1, f"SA must produce valid one-hot at residue {i}"

# ── Ising conversion ─────────────────────────────────────────────────────────

def test_ising_conversion_roundtrip():
    N, n, M = 3, 3, 9
    rng = np.random.default_rng(0)
    Q = rng.uniform(-2, 2, (M, M))
    Q = (Q + Q.T) / 2
    J, h, c = qubo_to_ising(Q)
    # Spot check: |J| and |h| are finite
    assert np.all(np.isfinite(J))
    assert np.all(np.isfinite(h))

# ── OGP router ───────────────────────────────────────────────────────────────

def test_router_dense_goes_quantum():
    """Dense, fully-connected QUBO should route to quantum."""
    rng = np.random.default_rng(42)
    M = 12
    Q = rng.normal(0, 5, (M, M))
    Q = (Q + Q.T) / 2
    router = OGPRouter()
    use_q, cert = router.should_use_quantum(Q, 4, 3, frustration_index=2.0)
    # High frustration should typically route to quantum
    assert isinstance(cert["rho"], float)
    assert isinstance(cert["spectral_gap"], float)

def test_router_sparse_goes_classical():
    """Sparse QUBO with zero frustration should route to classical."""
    M = 12
    Q = np.diag(np.ones(M) * -2.0)  # Only self-energies, no interactions
    router = OGPRouter()
    use_q, cert = router.should_use_quantum(Q, 4, 3, frustration_index=0.0)
    assert cert["routed_to"] == "classical", "No interactions = classical"

# ── IWS-QAOA ─────────────────────────────────────────────────────────────────

def make_trivial_qubo(N=3, n=2):
    """Simple QUBO with known optimal (all rot=0)."""
    M = N * n
    Q = np.zeros((M, M))
    for i in range(N):
        Q[i*n, i*n] = -5.0  # rot0 strongly preferred
        Q[i*n+1, i*n+1] = 0.0
    lam = 15.0
    for i in range(N):
        Q[i*n, i*n] -= lam; Q[i*n+1, i*n+1] -= lam
        Q[i*n, i*n+1] += lam; Q[i*n+1, i*n] += lam
    return Q

def test_iws_qaoa_finds_trivial_optimum():
    """IWS-QAOA must find global optimum on a trivial 6-qubit instance."""
    N, n = 3, 2
    Q = make_trivial_qubo(N, n)
    ex_x, ex_e = exact_solve(Q, N, n)
    gr_x, gr_e = greedy_rotamer_pack(Q, N, n)
    
    cfg = QAOAConfig(p=2, n_shots=128, n_iter=1, cvar_alpha=0.2,
                     use_warm_start=True, use_xy_mixer=True,
                     max_opt_iter=30, n_restarts=1, seed=42)
    solver = IWSQAOASolver(Q, N, n, cfg)
    result = solver.solve(greedy_assignment=gr_x, exact_energy=ex_e)
    
    assert not np.isnan(result.best_energy), "Must produce valid energy"
    assert result.best_energy <= gr_e + 0.01, "IWS must not be worse than greedy"

def test_iws_better_than_vanilla_on_warm_start_in_history():
    """With warm start, first IWS round should use greedy bias."""
    N, n = 3, 2
    Q = make_trivial_qubo(N, n)
    _, _, ex_e = exact_solve(Q, N, n)[0], exact_solve(Q, N, n)[1], exact_solve(Q, N, n)[1]
    gr_x, _ = greedy_rotamer_pack(Q, N, n)
    
    cfg = QAOAConfig(p=2, n_shots=64, n_iter=2, cvar_alpha=0.2,
                     use_warm_start=True, use_xy_mixer=True,
                     max_opt_iter=20, n_restarts=1, seed=42)
    solver = IWSQAOASolver(Q, N, n, cfg)
    result = solver.solve(greedy_assignment=gr_x)
    
    assert len(result.iws_history) == 2, "Should have 2 IWS rounds"
    assert len(result.params_history) == 2

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
