#!/usr/bin/env python3
"""
Phase 5 — QUBO/Ising Encoding
QADF Project: Hybrid Quantum-Classical Protein Structure Prediction
Target: Side-Chain Rotamer Optimization

This script encodes a 4-residue window from 1L2Y (residues 3-6)
as a Quadratic Unconstrained Binary Optimization (QUBO) problem.

ENCODING DECISIONS:
- One-hot encoding: 3 binary vars per residue (g-, t, g+)
  Reason: One-hot is the standard approach for discrete choice variables
  in QUBO formulations (Agathangelou et al. 2025, REF-02). It guarantees
  valid constraint checking via penalty terms and maps naturally to
  the Ising Hamiltonian for quantum hardware.
  Overhead: 3× more variables than log-encoding, but simpler penalty structure.
  
- 3 bins per chi1 angle: g- (gauche minus, ~-60°), t (trans, ~180°), g+ (gauche plus, ~60°)
  Reason: Covers >90% of rotamer population in Dunbrack library (REF-05).
  Coarser binning loses resolution but keeps qubit count tractable.
  
- 4-residue window: 4 residues × 3 states = 12 binary variables
  Reason: 12 qubits is within NISQ simulation range.
  5 residues → 15 qubits (marginal), 8 residues → 24 qubits (still OK for simulation),
  20 residues → 60 qubits (classical simulation feasible but slow, QPU not yet).

FEASIBILITY:
  n residues × 3 states = 3n binary variables
  n=4: 12 vars → tractable (exhaustive: 3^4 = 81 states)
  n=8: 24 vars → QUBO tractable, QPU marginal  
  n=10: 30 vars → QUBO tractable, classical simulation strained
  n=20: 60 vars → infeasible for NISQ QPU; slow for classical sim
  At 20-25 qubits or beyond: document as resource boundary
"""

import os
import sys
import json
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE_DIR = "/home/user/workspace/marena-qadf"
QUBO_DIR = os.path.join(BASE_DIR, "data/qubo")
ROTAMER_DIR = os.path.join(BASE_DIR, "data/rotamers")
os.makedirs(QUBO_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 5: QUBO/Ising Encoding")
print("=" * 60)

# ============================================================
# Load 1L2Y rotamer data
# ============================================================
df = pd.read_csv(os.path.join(ROTAMER_DIR, "1L2Y_rotamers.csv"))
print(f"\n1L2Y residues loaded: {len(df)}")
print(df[['res_seq', 'res_name', 'aa1', 'chi1', 'chi1_bin', 'ca_x', 'ca_y', 'ca_z']].to_string(index=False))

# ============================================================
# Select 4-residue window: residues 3–6 (0-indexed: 2–5)
# ============================================================
# 1L2Y sequence: NLYIQWLKDGGPSSGRPPPS
# Res 3-6 (1-indexed): Y(3), I(4), Q(5), W(6) 
# These are res_seq 3,4,5,6 in the PDB
window_mask = df['res_seq'].isin([3, 4, 5, 6])
window_df = df[window_mask].reset_index(drop=True)

print(f"\n--- 4-Residue Window (res 3-6): YIQW ---")
print(window_df[['res_seq', 'res_name', 'aa1', 'chi1', 'chi1_bin', 
                  'ca_x', 'ca_y', 'ca_z']].to_string(index=False))

n_residues = len(window_df)
n_states = 3  # g-, t, g+
n_vars = n_residues * n_states  # = 12
STATE_NAMES = ['g-', 't', 'g+']
STATE_ANGLES = np.array([-60.0, 180.0, 60.0])  # centroids in degrees

print(f"\nEncoding: {n_residues} residues × {n_states} states = {n_vars} binary variables")
print(f"State names: {STATE_NAMES}")
print(f"State centroids (chi1 deg): {STATE_ANGLES}")

# Variable naming: x_{i}_{k} = 1 if residue i is in state k
# Flattened index: idx(i, k) = i * n_states + k
def var_idx(i, k):
    return i * n_states + k

# ============================================================
# Ground truth rotamer assignment
# ============================================================
ground_truth = []
for i, row in window_df.iterrows():
    bin_name = row['chi1_bin']
    if bin_name == 'g-':
        gt_state = 0
    elif bin_name == 't':
        gt_state = 1
    elif bin_name == 'g+':
        gt_state = 2
    else:
        # No chi1 (Gly, Ala) — assign t as default
        gt_state = 1
    ground_truth.append(gt_state)
    print(f"  Residue {row['res_seq']} ({row['res_name']}): chi1={row['chi1']}, bin={bin_name} → state {gt_state} ({STATE_NAMES[gt_state]})")

ground_truth = np.array(ground_truth)

# ============================================================
# Self-energy (single-body) terms
# ============================================================
# Eself(i, k) = -log(p(rotamer_k | backbone_i)) from Dunbrack library
# We approximate this using:
# 1. Dunbrack probability prior: use simplified table
# 2. Deviation penalty: (chi1_k_centroid - chi1_actual_i)^2 / sigma^2
# Note: For residues without chi1, all states have equal prior

# Simplified Dunbrack prior probabilities (approximate, by amino acid type)
# From: Dunbrack 2011, DOI: 10.1002/prot.22921 (REF-05)
# Typical values for common amino acids in average backbone context
DUNBRACK_PRIOR = {
    'TYR': [0.40, 0.20, 0.40],  # g-, t, g+
    'ILE': [0.50, 0.15, 0.35],
    'GLN': [0.35, 0.30, 0.35],
    'TRP': [0.40, 0.25, 0.35],
    'DEFAULT': [0.33, 0.34, 0.33],  # uniform prior for unknown
}

def get_self_energy(res_name, state_idx, actual_chi1=None):
    """
    Compute self-energy for residue res_name in rotamer state state_idx.
    
    E_self = E_dunbrack + E_deviation
    
    E_dunbrack = -log(p(state | backbone)) using simplified prior
    E_deviation = (chi1_centroid - chi1_actual)^2 / (2 * sigma^2)
      where sigma = 20 degrees (typical rotamer well width)
    
    WHY: The Dunbrack term encodes statistical preferences from protein database.
         The deviation term penalizes states far from the observed chi1 angle.
         Together they approximate the true self-energy in the QUBO formulation.
    """
    prior = DUNBRACK_PRIOR.get(res_name, DUNBRACK_PRIOR['DEFAULT'])
    e_dunbrack = -math.log(prior[state_idx] + 1e-10)
    
    e_deviation = 0.0
    if actual_chi1 is not None:
        centroid = STATE_ANGLES[state_idx]
        # Handle periodicity: use minimum angular distance
        diff = actual_chi1 - centroid
        diff = ((diff + 180) % 360) - 180  # wrap to [-180, 180]
        sigma = 20.0  # degrees
        e_deviation = (diff ** 2) / (2 * sigma ** 2)
    
    return e_dunbrack + e_deviation

# ============================================================
# Pairwise interaction energies (two-body terms)
# ============================================================
# Use simplified Lennard-Jones-inspired energy:
# E_pair(i,k, j,l) = epsilon_ij * [(r_ij_vdw / d_ij_kl)^12 - 2*(r_ij_vdw / d_ij_kl)^6]
# 
# WHERE:
# d_ij_kl = distance between CB atoms of residues i and j when
#            residue i is in state k and residue j is in state l.
#
# We approximate CB position from CA coordinates using the chi1 centroid:
# CB is displaced from CA by a bond length (~1.52 Å) in a direction
# determined by the backbone geometry (simplified: use random perturbation
# based on chi1 angle, since we don't have full backbone geometry).
#
# WHY THIS SIMPLIFICATION: True pairwise energies require all side-chain
# atom positions. For a proof-of-concept QUBO, we use CB as a proxy
# and a simplified LJ potential. Full implementation would use PyRosetta
# (as in Agathangelou et al. 2025, REF-02) for accurate pairwise energies.
#
# WHY LJ POTENTIAL: Van der Waals interactions dominate at the short
# distances relevant to steric clashes between side chains (2-4 Å).
# The LJ 12-6 form captures repulsion (short range) and attraction (medium range).

VDW_RADII = {  # Approximate CB VDW radius in Angstroms
    'TYR': 1.9, 'ILE': 1.8, 'GLN': 1.7, 'TRP': 2.0, 'DEFAULT': 1.7
}

def estimate_cb_position(ca_coords, res_seq, state_k):
    """
    Estimate CB atom position from CA coordinates.
    
    Simplified: CB is at distance 1.52 Å from CA.
    Direction is approximated as a function of chi1 centroid angle.
    We use a fixed reference direction (along z) rotated by chi1.
    
    This is a geometric approximation — real CB position depends on
    the full backbone geometry (N-CA-CB angle = ~110°).
    """
    ca = np.array([ca_coords['x'], ca_coords['y'], ca_coords['z']])
    chi1_rad = math.radians(STATE_ANGLES[state_k])
    
    # Simplified CB offset: rotate a unit vector in the xy plane
    # by the chi1 angle. This is a 2D approximation of the real 3D geometry.
    bond_length = 1.52  # Angstroms, C-C bond
    cb_offset = bond_length * np.array([
        math.cos(chi1_rad),
        math.sin(chi1_rad),
        0.3  # small z component (approximate out-of-plane geometry)
    ])
    cb_offset /= np.linalg.norm(cb_offset)
    cb_offset *= bond_length
    
    return ca + cb_offset

def lj_energy(d, r_vdw_sum, epsilon=0.5):
    """
    Simplified Lennard-Jones energy.
    E = epsilon * [(r_vdw_sum/d)^12 - 2*(r_vdw_sum/d)^6]
    
    Minimum at d = r_vdw_sum (equilibrium distance).
    Positive (repulsive) for d < r_vdw_sum.
    Negative (attractive) for d > r_vdw_sum.
    Capped to avoid extreme values from very small distances.
    """
    if d < 0.1:  # avoid division by zero
        return 10.0  # large repulsion
    ratio = r_vdw_sum / d
    ratio6 = ratio ** 6
    e = epsilon * (ratio6 ** 2 - 2 * ratio6)
    return max(min(e, 10.0), -2.0)  # cap for numerical stability

def get_pair_energy(res_i, row_i, state_k, res_j, row_j, state_l):
    """
    Compute pairwise interaction energy between residue i in state k
    and residue j in state l.
    """
    ca_i = {'x': row_i['ca_x'], 'y': row_i['ca_y'], 'z': row_i['ca_z']}
    ca_j = {'x': row_j['ca_x'], 'y': row_j['ca_y'], 'z': row_j['ca_z']}
    
    cb_i = estimate_cb_position(ca_i, row_i['res_seq'], state_k)
    cb_j = estimate_cb_position(ca_j, row_j['res_seq'], state_l)
    
    d = np.linalg.norm(cb_i - cb_j)
    
    r_i = VDW_RADII.get(res_i, VDW_RADII['DEFAULT'])
    r_j = VDW_RADII.get(res_j, VDW_RADII['DEFAULT'])
    r_sum = r_i + r_j
    
    return lj_energy(d, r_sum)

# ============================================================
# Build 12×12 QUBO matrix Q
# ============================================================
print("\n--- Building 12×12 QUBO Matrix ---")

P = 10.0  # One-hot penalty coefficient
# WHY P=10: The penalty P * (sum_i x_i - 1)^2 must be strong enough
# to enforce the one-hot constraint relative to the energy scale.
# With self-energies typically in [0, 10] range, P=10 ensures constraint
# violations are always unfavorable. Too large P causes numerical issues.

Q = np.zeros((n_vars, n_vars))

# DIAGONAL: self-energy + one-hot contribution
for i in range(n_residues):
    row_i = window_df.iloc[i]
    res_name = row_i['res_name']
    actual_chi1 = row_i['chi1'] if not pd.isna(row_i['chi1']) else None
    
    for k in range(n_states):
        idx = var_idx(i, k)
        # Self-energy term
        e_self = get_self_energy(res_name, k, actual_chi1)
        # One-hot diagonal: P * (x_{ik}^2 - 2*x_{ik} + ...)
        # Expanded: P * x_{ik}^2 - 2P * x_{ik}  (with cross terms in off-diagonal)
        # Since x^2 = x for binary, diagonal gets: e_self - P
        Q[idx, idx] += e_self - P  # -P from one-hot expansion

# ONE-HOT CROSS TERMS (within same residue, different states)
for i in range(n_residues):
    for k1 in range(n_states):
        for k2 in range(k1 + 1, n_states):
            idx1 = var_idx(i, k1)
            idx2 = var_idx(i, k2)
            # Penalty for selecting two states for same residue
            Q[idx1, idx2] += 2 * P  # upper triangle
            # WHY 2P: Expansion of P*(sum_k x_k - 1)^2 produces
            # off-diagonal terms P * 2 * x_k1 * x_k2

# PAIRWISE INTERACTION TERMS (between different residues)
for i in range(n_residues):
    for j in range(i + 1, n_residues):
        row_i = window_df.iloc[i]
        row_j = window_df.iloc[j]
        
        for k in range(n_states):
            for l in range(n_states):
                idx_ik = var_idx(i, k)
                idx_jl = var_idx(j, l)
                
                e_pair = get_pair_energy(
                    row_i['res_name'], row_i,
                    k,
                    row_j['res_name'], row_j,
                    l
                )
                
                # Upper triangle only (QUBO convention)
                if idx_ik < idx_jl:
                    Q[idx_ik, idx_jl] += e_pair
                else:
                    Q[idx_jl, idx_ik] += e_pair

print(f"QUBO matrix shape: {Q.shape}")
print(f"QUBO matrix stats:")
print(f"  Min: {Q.min():.4f}")
print(f"  Max: {Q.max():.4f}")
print(f"  Mean (off-diag): {Q[Q != 0].mean():.4f}")
print(f"  Non-zero elements: {np.count_nonzero(Q)}")

# ============================================================
# Verify: energy for ground-truth assignment
# ============================================================
def compute_qubo_energy(x, Q):
    """Compute QUBO energy E = x^T Q x for binary vector x."""
    return float(x @ Q @ x)

def state_to_binary(states, n_residues, n_states):
    """Convert state assignment list to binary one-hot vector."""
    x = np.zeros(n_residues * n_states, dtype=float)
    for i, s in enumerate(states):
        x[var_idx(i, s)] = 1.0
    return x

# Ground truth binary encoding
x_gt = state_to_binary(ground_truth, n_residues, n_states)
E_gt = compute_qubo_energy(x_gt, Q)
print(f"\nGround truth assignment: {[STATE_NAMES[s] for s in ground_truth]}")
print(f"Ground truth binary: {x_gt.astype(int)}")
print(f"Ground truth QUBO energy: {E_gt:.4f}")

# Enumerate all valid configurations (3^4 = 81) to check ground truth rank
print("\n--- Enumerating all 81 valid configurations ---")
all_energies = []
all_states = []

from itertools import product
for states in product(range(n_states), repeat=n_residues):
    x = state_to_binary(states, n_residues, n_states)
    E = compute_qubo_energy(x, Q)
    all_energies.append(E)
    all_states.append(states)

all_energies = np.array(all_energies)
sorted_idx = np.argsort(all_energies)

# Find ground truth rank
gt_state_tuple = tuple(ground_truth)
gt_rank = np.where(np.array(all_states)[sorted_idx] == gt_state_tuple)[0]
# Alternative: find by energy match
gt_rank_by_energy = np.searchsorted(np.sort(all_energies), E_gt)

print(f"Minimum energy configuration: {[STATE_NAMES[s] for s in all_states[sorted_idx[0]]]}")
print(f"Minimum energy value: {all_energies[sorted_idx[0]]:.4f}")
print(f"Ground truth energy: {E_gt:.4f}")
print(f"Ground truth rank (among 81): {gt_rank_by_energy + 1} / 81")
print(f"  (lower rank = closer to optimal solution)")

# Is ground truth among the lower-energy configurations?
pct_rank = (gt_rank_by_energy + 1) / 81 * 100
print(f"Ground truth percentile: {pct_rank:.1f}% (lower = better)")
if pct_rank <= 50:
    print("  ✓ Ground truth is in the lower half of configurations (as expected)")
else:
    print("  Note: Ground truth not in lower half — check encoding")

# Top 10 configurations
print("\nTop 10 lowest-energy configurations:")
print(f"{'Rank':<6} {'States':<20} {'Energy':<10}")
print("-" * 40)
for rank in range(min(10, len(sorted_idx))):
    idx = sorted_idx[rank]
    states = all_states[idx]
    E = all_energies[idx]
    state_str = ','.join([STATE_NAMES[s] for s in states])
    marker = " ← GT" if states == gt_state_tuple else ""
    print(f"{rank+1:<6} {state_str:<20} {E:<10.4f}{marker}")

# ============================================================
# Save QUBO matrix and metadata
# ============================================================
qubo_path = os.path.join(QUBO_DIR, "qubo_matrix.npy")
np.save(qubo_path, Q)
print(f"\nQUBO matrix saved: {qubo_path}")

# Save all energies for use in Phase 6
energies_data = {
    "all_energies": all_energies.tolist(),
    "all_states": [list(s) for s in all_states],
    "sorted_indices": sorted_idx.tolist(),
    "ground_truth_states": ground_truth.tolist(),
    "ground_truth_energy": float(E_gt),
    "min_energy": float(all_energies.min()),
    "ground_truth_rank": int(gt_rank_by_energy + 1),
}
energies_path = os.path.join(QUBO_DIR, "all_energies.json")
with open(energies_path, 'w') as f:
    json.dump(energies_data, f, indent=2)

metadata = {
    "n_residues": n_residues,
    "n_states_per_residue": n_states,
    "n_binary_variables": n_vars,
    "residues": [
        {
            "index": int(i),
            "res_seq": int(window_df.iloc[i]['res_seq']),
            "res_name": str(window_df.iloc[i]['res_name']),
            "aa1": str(window_df.iloc[i]['aa1']),
            "chi1_actual": float(window_df.iloc[i]['chi1']) if not pd.isna(window_df.iloc[i]['chi1']) else None,
            "chi1_bin": str(window_df.iloc[i]['chi1_bin']) if not pd.isna(window_df.iloc[i]['chi1_bin']) else None,
            "ground_truth_state": int(ground_truth[i]),
        }
        for i in range(n_residues)
    ],
    "state_names": STATE_NAMES,
    "state_centroids_deg": STATE_ANGLES.tolist(),
    "one_hot_penalty_P": P,
    "encoding_decisions": {
        "one_hot": "Standard QUBO approach (Agathangelou et al. 2025, REF-02); enables constraint checking via penalty terms",
        "3_bins": "Covers >90% of Dunbrack rotamer population; tractable qubit count (3n qubits for n residues)",
        "4_residue_window": "12 qubits = NISQ-feasible; 3^4=81 states = exhaustively enumerable",
        "lj_potential": "Approximate pairwise energy; real implementation would use PyRosetta scoring (REF-02)",
        "feasibility": {
            "n=4": "12 vars, 81 states: tractable exhaustively",
            "n=6": "18 vars, 729 states: tractable classically",
            "n=8": "24 vars, 6561 states: tractable classically",
            "n=10": "30 vars, ~59k states: classical strained",
            "n=20": "60 vars, ~3.5B states: quantum sim feasible but slow; QPU target",
            "infeasibility_boundary": "~20-25 qubits for NISQ QPU; ~30-35 qubits for classical statevector sim"
        }
    },
    "qubo_stats": {
        "min": float(Q.min()),
        "max": float(Q.max()),
        "n_nonzero": int(np.count_nonzero(Q)),
        "ground_truth_energy": float(E_gt),
        "min_energy": float(all_energies.min()),
        "ground_truth_rank_of_81": int(gt_rank_by_energy + 1)
    }
}

metadata_path = os.path.join(QUBO_DIR, "encoding_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Encoding metadata saved: {metadata_path}")
print("\nPhase 5 complete.")
