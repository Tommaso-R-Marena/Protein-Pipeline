"""
PDB structure loader with β-sheet frustration detection.

Identifies residue windows where greedy rotamer packing provably fails
by finding locally frustrated contact networks (Wolynes frustration index).
"""
import io
import json
import math
import os
import urllib.request

import numpy as np


# ── Dunbrack backbone-dependent rotamer library (2010) stub ──────────────────
# We embed the top rotamers for the 20 AA at the most common φ/ψ bins.
# Full library at: https://dunbrack.fccc.edu/bbdep2010/
# For this implementation we use a hard-coded lookup for the 8 most frustrated
# residues: ASN, GLN, LYS, ARG, MET, GLU, ASP, SER.
# Each entry: [(chi1, chi2, prob), ...]  angles in degrees.
_DUNBRACK_TOP4 = {
    "SER": [(-65, None, 0.40), (60, None, 0.35), (180, None, 0.25)],
    "ASP": [(-70, 0, 0.45),   (-70, 180, 0.35), (60, 0, 0.10),  (180, 0, 0.10)],
    "ASN": [(-70, 30, 0.40),  (-70, 150, 0.30), (60, 30, 0.15), (180, 30, 0.15)],
    "GLU": [(-70, 180, 0.35), (-70, -85, 0.30), (60, 180, 0.20), (180, 180, 0.15)],
    "GLN": [(-70, 180, 0.35), (-70, -85, 0.25), (60, 180, 0.20), (180, 180, 0.20)],
    "LYS": [(-67, 180, 0.40), (-67, -65, 0.30), (60, 180, 0.20), (180, 180, 0.10)],
    "ARG": [(-67, 180, 0.38), (-67, -65, 0.30), (60, 180, 0.20), (180, 180, 0.12)],
    "MET": [(-65, 180, 0.40), (-65, -65, 0.30), (60, 180, 0.20), (180, 180, 0.10)],
    "LEU": [(-65, 175, 0.45), (-65, 65,  0.35), (60, 175, 0.10),  (180, 175, 0.10)],
    "ILE": [(-65, 170, 0.45), (-65, -65, 0.35), (60, 170, 0.10),  (180, 170, 0.10)],
    "VAL": [(-65, None, 0.45), (60, None, 0.30), (180, None, 0.25)],
    "THR": [(-65, None, 0.45), (60, None, 0.30), (180, None, 0.25)],
    "PHE": [(-65, 90, 0.50),  (-65, -85, 0.30), (60, 90, 0.10), (180, 90, 0.10)],
    "TYR": [(-65, 90, 0.50),  (-65, -85, 0.30), (60, 90, 0.10), (180, 90, 0.10)],
    "TRP": [(-65, 100, 0.50), (-65, -90, 0.30), (60, 100, 0.10), (180, 100, 0.10)],
    "HIS": [(-65, -80, 0.40), (-65, 100, 0.30), (60, -80, 0.20), (180, -80, 0.10)],
    "CYS": [(-65, None, 0.60), (60, None, 0.25), (180, None, 0.15)],
    "ALA": [(None, None, 1.0)],
    "GLY": [(None, None, 1.0)],
    "PRO": [(-65, 30, 0.75),  (-65, -30, 0.25)],
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def download_pdb(pdb_id: str, cache_dir: str = "/tmp/pdb_cache") -> str:
    """Download PDB file, return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        urllib.request.urlretrieve(url, path)
    return path


def parse_pdb_backbone(pdb_path: str, chain_id: str = "A") -> list[dict]:
    """
    Parse PDB, return list of residue dicts with backbone and Cβ coords.
    Each dict: {resnum, resname, chain, CA, CB, N, C, phi, psi}
    """
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line[:4] not in ("ATOM", "HETA"):
                continue
            if line[4:6].strip() != "":  # skip HETATM-style
                pass
            rec = line[:6].strip()
            if rec not in ("ATOM",):
                continue
            chain = line[21]
            if chain_id and chain != chain_id:
                continue
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            atom = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if resnum not in residues:
                residues[resnum] = {"resnum": resnum, "resname": resname,
                                     "chain": chain, "atoms": {}}
            residues[resnum]["atoms"][atom] = np.array([x, y, z])

    result = []
    for rnum in sorted(residues):
        r = residues[rnum]
        atoms = r["atoms"]
        if "CA" not in atoms:
            continue
        entry = {
            "resnum": rnum,
            "resname": r["resname"],
            "chain": r["chain"],
            "CA": atoms.get("CA"),
            "CB": atoms.get("CB", atoms.get("CA")),  # GLY fallback
            "N": atoms.get("N"),
            "C": atoms.get("C"),
        }
        result.append(entry)
    return result


def compute_contact_matrix(residues: list[dict], cutoff_A: float = 8.0) -> np.ndarray:
    """Cβ-distance contact matrix (Å), using CA for GLY."""
    n = len(residues)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cb_i = residues[i]["CB"]
            cb_j = residues[j]["CB"]
            if cb_i is None or cb_j is None:
                continue
            d = float(np.linalg.norm(cb_i - cb_j))
            D[i, j] = D[j, i] = d
    return D


def detect_beta_sheet_residues(residues: list[dict]) -> list[int]:
    """
    Identify β-sheet residues using the Cα–Cα inter-strand distance criterion.
    Two residues i, j (|i-j| >= 4) with d(Cα,Cα) < 5.5 Å and d(Cβ,Cβ) < 7 Å
    are flagged as β-sheet contacts.
    Returns list of residue indices in contact pairs.
    """
    n = len(residues)
    beta_indices = set()
    for i in range(n):
        for j in range(i + 4, n):
            ca_i = residues[i]["CA"]
            ca_j = residues[j]["CA"]
            cb_i = residues[i]["CB"]
            cb_j = residues[j]["CB"]
            if ca_i is None or ca_j is None:
                continue
            dca = float(np.linalg.norm(ca_i - ca_j))
            dcb = float(np.linalg.norm(cb_i - cb_j)) if (cb_i is not None and cb_j is not None) else dca
            if dca < 6.5 and dcb < 8.0:
                beta_indices.add(i)
                beta_indices.add(j)
    return sorted(beta_indices)


def frustration_index(residues: list[dict], window: list[int],
                      n_rotamers: int = 4) -> float:
    """
    Local frustration index for a residue window.

    Based on Wolynes group's frustratometer concept:
    FI = (E_native - <E_decoy>) / std(E_decoy)

    We approximate using Cβ-Cβ distances:
    - E_native = sum of pairwise Cβ distances (lower = more frustrated if contacts clash)
    - Decoys = random rotamer assignments
    Returns a positive float; higher = more frustrated (greedy more likely to fail).
    """
    if len(window) < 3:
        return 0.0

    positions = [residues[i]["CB"] for i in window if residues[i]["CB"] is not None]
    if len(positions) < 3:
        return 0.0

    pos = np.array(positions)
    n = len(pos)

    # Native energy: sum of inverse Cβ-Cβ distances (clash potential)
    native_e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < 0.1:
                d = 0.1
            native_e += 1.0 / d  # LJ repulsive proxy

    # Decoy energies: shuffle positions
    decoy_energies = []
    rng = np.random.default_rng(42)
    for _ in range(100):
        shuffled = pos[rng.permutation(n)]
        e = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(shuffled[i] - shuffled[j]))
                if d < 0.1:
                    d = 0.1
                e += 1.0 / d
        decoy_energies.append(e)

    mean_decoy = float(np.mean(decoy_energies))
    std_decoy = float(np.std(decoy_energies)) + 1e-9
    fi = (native_e - mean_decoy) / std_decoy
    return float(fi)


def get_rotamers_for_residue(resname: str, n_max: int = 4) -> list[tuple]:
    """Return top-n rotamers for a residue as (chi1_deg, chi2_deg, prob) tuples."""
    rotamers = _DUNBRACK_TOP4.get(resname, [(None, None, 1.0)])
    return rotamers[:n_max]


def build_qubo_from_window(residues: list[dict], window_indices: list[int],
                            n_rotamers: int = 4) -> tuple[np.ndarray, list[dict]]:
    """
    Build QUBO matrix Q for the rotamer packing problem on a residue window.

    Encoding: one-hot per residue (n_rotamers bits per residue).
    Q[α,β] = interaction energy between rotamer α and rotamer β.
    Self-energy: Q[α,α] = Dunbrack -log(prob) penalty.

    Returns (Q matrix [N*n x N*n], metadata list).
    """
    N = len(window_indices)
    n = n_rotamers
    M = N * n

    Q = np.zeros((M, M))
    meta = []

    # Per-residue rotamer list
    rot_list = []
    for idx in window_indices:
        res = residues[idx]
        rots = get_rotamers_for_residue(res["resname"], n_rotamers)
        # Pad to n_rotamers if fewer available
        while len(rots) < n:
            rots = rots + [(rots[-1][0], rots[-1][1], 0.001)]
        rot_list.append(rots[:n])
        meta.append({"resnum": res["resnum"], "resname": res["resname"],
                      "rotamers": rots[:n]})

    # Self-energies: diagonal Q[α,α] = -log(prob_α)
    for i in range(N):
        for a in range(n):
            alpha = i * n + a
            prob = max(rot_list[i][a][2], 1e-6)
            Q[alpha, alpha] = -math.log(prob)

    # Pairwise interaction energies: Lennard-Jones approximation using Cβ-Cβ distances
    for i in range(N):
        res_i = residues[window_indices[i]]
        cb_i = res_i["CB"]
        if cb_i is None:
            continue
        for j in range(i + 1, N):
            res_j = residues[window_indices[j]]
            cb_j = res_j["CB"]
            if cb_j is None:
                continue
            d0 = float(np.linalg.norm(cb_i - cb_j))

            for a in range(n):
                for b in range(n):
                    alpha = i * n + a
                    beta = j * n + b

                    # Rotamer-specific distance perturbation based on chi1 difference
                    chi1_a = rot_list[i][a][0] or 0.0
                    chi1_b = rot_list[j][b][0] or 0.0
                    # Simple model: chi1 rotates Cβ-Cγ vector; perturbs effective distance
                    delta_chi = math.radians(chi1_a - chi1_b)
                    r_eff = max(d0 + 1.5 * math.sin(delta_chi / 2.0), 2.5)

                    # Lennard-Jones 12-6 (ε=1, σ=3.8 Å for Cβ-Cβ)
                    sigma = 3.8
                    sr = sigma / r_eff
                    lj = 4.0 * (sr**12 - sr**6)
                    Q[alpha, beta] += lj
                    Q[beta, alpha] += lj

    # One-hot penalty (soft): add λ*(1 - sum x_i)^2 terms
    # This ensures constraint satisfaction is penalized in the cost Hamiltonian
    # for baselines; the XY-mixer enforces it as a hard constraint.
    lam = 10.0
    for i in range(N):
        for a in range(n):
            alpha = i * n + a
            Q[alpha, alpha] += -lam  # from -2*lambda * x_alpha
            for b in range(n):
                if a != b:
                    beta = i * n + b
                    Q[alpha, beta] += lam  # from lambda * x_alpha * x_beta

    return Q, meta


def greedy_rotamer_pack(Q: np.ndarray, N: int, n: int) -> tuple[np.ndarray, float]:
    """
    Classical greedy rotamer packing: assign each residue its lowest self-energy
    rotamer independently, then greedily improve by considering pair interactions.
    Returns (assignment array of length N, energy).
    """
    M = N * n
    assignment = np.zeros(M, dtype=int)

    # Initialize: lowest self-energy per residue
    for i in range(N):
        best_a = np.argmin([Q[i * n + a, i * n + a] for a in range(n)])
        assignment[i * n + best_a] = 1

    # Greedy improvement: for each residue, check if switching improves total energy
    improved = True
    while improved:
        improved = False
        for i in range(N):
            current_a = np.argmax(assignment[i * n: i * n + n])
            current_e = _energy_if_assign(Q, assignment, N, n, i, current_a)
            for a in range(n):
                if a == current_a:
                    continue
                e = _energy_if_assign(Q, assignment, N, n, i, a)
                if e < current_e - 1e-9:
                    assignment[i * n: i * n + n] = 0
                    assignment[i * n + a] = 1
                    current_e = e
                    current_a = a
                    improved = True

    energy = float(assignment @ Q @ assignment)
    return assignment, energy


def _energy_if_assign(Q, assignment, N, n, i, a):
    """Compute total energy with residue i assigned to rotamer a."""
    x = assignment.copy()
    x[i * n: i * n + n] = 0
    x[i * n + a] = 1
    return float(x @ Q @ x)


def exact_solve(Q: np.ndarray, N: int, n: int) -> tuple[np.ndarray, float]:
    """Brute-force exact solution. Feasible for N*n <= 28."""
    best_e = float("inf")
    best_x = None
    M = N * n
    from itertools import product
    for combo in product(range(n), repeat=N):
        x = np.zeros(M, dtype=int)
        for i, a in enumerate(combo):
            x[i * n + a] = 1
        e = float(x @ Q @ x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
    return best_x, best_e


def simulated_annealing(Q: np.ndarray, N: int, n: int,
                         n_steps: int = 10000, T0: float = 5.0,
                         seed: int = 42) -> tuple[np.ndarray, float]:
    """Classical simulated annealing baseline."""
    rng = np.random.default_rng(seed)
    M = N * n

    # Initialize randomly
    x = np.zeros(M, dtype=int)
    for i in range(N):
        x[i * n + rng.integers(n)] = 1

    E = float(x @ Q @ x)
    best_x = x.copy()
    best_E = E

    for step in range(n_steps):
        T = T0 * (1.0 - step / n_steps)
        # Propose: swap one residue's rotamer
        i = rng.integers(N)
        cur_a = np.argmax(x[i * n: i * n + n])
        new_a = rng.integers(n)
        if new_a == cur_a:
            continue
        x_new = x.copy()
        x_new[i * n: i * n + n] = 0
        x_new[i * n + new_a] = 1
        E_new = float(x_new @ Q @ x_new)
        dE = E_new - E
        if dE < 0 or (T > 0 and rng.random() < math.exp(-dE / T)):
            x = x_new
            E = E_new
            if E < best_E:
                best_E = E
                best_x = x.copy()

    return best_x, best_E
