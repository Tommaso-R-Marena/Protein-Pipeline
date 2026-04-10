"""
PDB structure parsing and geometric utilities.

Handles downloading PDB files, extracting backbone coordinates,
computing torsion angles, distance matrices, and structural features.
"""

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    from Bio.PDB import PDBParser, PDBList
    from Bio.PDB.Polypeptide import protein_letters_3to1
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# Standard amino acid 3-letter to 1-letter mapping
AA3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# Standard backbone bond lengths (Angstroms)
BOND_LENGTHS = {
    ('N', 'CA'): 1.458,
    ('CA', 'C'): 1.524,
    ('C', 'N'): 1.329,
    ('C', 'O'): 1.231,
}

# Ramachandran regions (phi, psi) in degrees
RAMA_REGIONS = {
    'alpha_helix': ((-80, -40), (-60, -20)),
    'beta_sheet': ((-150, -90), (90, 170)),
    'left_helix': ((40, 80), (20, 60)),
    'polyproline': ((-80, -60), (120, 180)),
}


@dataclass
class ProteinStructure:
    """Parsed protein structure with backbone coordinates."""
    pdb_id: str
    chain_id: str
    sequence: str
    residue_numbers: np.ndarray          # (L,) int
    ca_coords: np.ndarray                # (L, 3) backbone CA coordinates
    n_coords: Optional[np.ndarray]       # (L, 3) backbone N
    c_coords: Optional[np.ndarray]       # (L, 3) backbone C
    cb_coords: Optional[np.ndarray]      # (L, 3) CB coordinates (None for GLY)
    phi_angles: Optional[np.ndarray]     # (L,) phi torsion angles in radians
    psi_angles: Optional[np.ndarray]     # (L,) psi torsion angles in radians
    omega_angles: Optional[np.ndarray]   # (L,) omega torsion angles
    bfactors: Optional[np.ndarray]       # (L,) B-factors

    @property
    def length(self) -> int:
        return len(self.sequence)

    def distance_matrix(self) -> np.ndarray:
        """Compute CA-CA distance matrix."""
        diff = self.ca_coords[:, None, :] - self.ca_coords[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def contact_map(self, threshold: float = 8.0) -> np.ndarray:
        """Binary contact map at given distance threshold."""
        return (self.distance_matrix() < threshold).astype(np.float32)

    def get_region(self, start: int, end: int) -> "ProteinStructure":
        """Extract a sub-region of the structure."""
        mask = (self.residue_numbers >= start) & (self.residue_numbers <= end)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No residues found in range [{start}, {end}]")
        return ProteinStructure(
            pdb_id=self.pdb_id,
            chain_id=self.chain_id,
            sequence="".join(self.sequence[i] for i in indices),
            residue_numbers=self.residue_numbers[indices],
            ca_coords=self.ca_coords[indices],
            n_coords=self.n_coords[indices] if self.n_coords is not None else None,
            c_coords=self.c_coords[indices] if self.c_coords is not None else None,
            cb_coords=self.cb_coords[indices] if self.cb_coords is not None else None,
            phi_angles=self.phi_angles[indices] if self.phi_angles is not None else None,
            psi_angles=self.psi_angles[indices] if self.psi_angles is not None else None,
            omega_angles=self.omega_angles[indices] if self.omega_angles is not None else None,
            bfactors=self.bfactors[indices] if self.bfactors is not None else None,
        )


def compute_dihedral(p0: np.ndarray, p1: np.ndarray,
                     p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle between four points in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return math.atan2(y, x)


def compute_backbone_torsions(
    n_coords: np.ndarray,
    ca_coords: np.ndarray,
    c_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute phi, psi, omega torsion angles for a chain.
    
    Returns:
        phi: (L,) array, phi[0] = NaN (undefined for first residue)
        psi: (L,) array, psi[-1] = NaN (undefined for last residue)
        omega: (L,) array, omega[0] = NaN
    """
    L = len(ca_coords)
    phi = np.full(L, np.nan)
    psi = np.full(L, np.nan)
    omega = np.full(L, np.nan)

    for i in range(1, L):
        # phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        phi[i] = compute_dihedral(c_coords[i-1], n_coords[i],
                                   ca_coords[i], c_coords[i])
    for i in range(L - 1):
        # psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])
        psi[i] = compute_dihedral(n_coords[i], ca_coords[i],
                                   c_coords[i], n_coords[i+1])
    for i in range(1, L):
        # omega[i] = dihedral(CA[i-1], C[i-1], N[i], CA[i])
        omega[i] = compute_dihedral(ca_coords[i-1], c_coords[i-1],
                                     n_coords[i], ca_coords[i])
    return phi, psi, omega


def discretize_angles(angles: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Discretize torsion angles into bins.
    
    Maps angles from [-pi, pi] to bin indices [0, num_bins-1].
    """
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    bins = np.digitize(angles, bin_edges) - 1
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def bin_centers(num_bins: int = 8) -> np.ndarray:
    """Get the center angle (radians) for each bin."""
    edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def parse_pdb(pdb_path: str, chain_id: str = "A",
              pdb_id: str = "XXXX") -> ProteinStructure:
    """Parse a PDB file and extract backbone information."""
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython is required for PDB parsing")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    if chain_id not in [c.id for c in model.get_chains()]:
        available = [c.id for c in model.get_chains()]
        raise ValueError(
            f"Chain {chain_id} not found. Available: {available}")

    chain = model[chain_id]

    seq_list = []
    res_nums = []
    ca_list, n_list, c_list, cb_list = [], [], [], []
    bf_list = []

    for residue in chain.get_residues():
        resname = residue.get_resname()
        if resname not in AA3TO1:
            continue
        hetfield = residue.get_id()[0]
        if hetfield != " ":
            continue

        atoms = {a.get_name(): a for a in residue.get_atoms()}
        if "CA" not in atoms or "N" not in atoms or "C" not in atoms:
            continue

        seq_list.append(AA3TO1[resname])
        res_nums.append(residue.get_id()[1])
        ca_list.append(atoms["CA"].get_vector().get_array())
        n_list.append(atoms["N"].get_vector().get_array())
        c_list.append(atoms["C"].get_vector().get_array())
        cb_list.append(
            atoms["CB"].get_vector().get_array() if "CB" in atoms else
            atoms["CA"].get_vector().get_array()  # GLY fallback
        )
        bf_list.append(atoms["CA"].get_bfactor())

    if not seq_list:
        raise ValueError(f"No standard residues found in chain {chain_id}")

    ca_coords = np.array(ca_list)
    n_coords = np.array(n_list)
    c_coords = np.array(c_list)
    cb_coords = np.array(cb_list)

    phi, psi, omega = compute_backbone_torsions(n_coords, ca_coords, c_coords)

    return ProteinStructure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence="".join(seq_list),
        residue_numbers=np.array(res_nums),
        ca_coords=ca_coords,
        n_coords=n_coords,
        c_coords=c_coords,
        cb_coords=cb_coords,
        phi_angles=phi,
        psi_angles=psi,
        omega_angles=omega,
        bfactors=np.array(bf_list),
    )


def download_pdb(pdb_id: str, save_dir: str = "data/pdb") -> str:
    """Download a PDB file from RCSB."""
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython required for PDB download")

    os.makedirs(save_dir, exist_ok=True)
    pdbl = PDBList()
    filename = pdbl.retrieve_pdb_file(
        pdb_id, file_format="pdb", pdir=save_dir, overwrite=False
    )
    return filename


def superimpose_structures(
    mobile: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Kabsch superimposition of two coordinate sets.
    
    Args:
        mobile: (N, 3) coordinates to transform
        target: (N, 3) reference coordinates
    
    Returns:
        transformed: (N, 3) superimposed mobile coordinates
        rmsd: RMSD after superimposition
    """
    assert mobile.shape == target.shape
    n = mobile.shape[0]

    # Center both
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Compute rotation matrix via SVD
    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Apply transformation
    transformed = (R @ mobile_centered.T).T + target_center

    # Compute RMSD
    diff = transformed - target
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return transformed, rmsd


def compute_gdt_ts(
    pred: np.ndarray, target: np.ndarray,
    thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute GDT-TS (Global Distance Test - Total Score)."""
    _, rmsd_per_residue = superimpose_structures(pred, target)
    # Per-residue distances after superimposition
    pred_sup, _ = superimpose_structures(pred, target)
    distances = np.sqrt(np.sum((pred_sup - target)**2, axis=1))
    
    scores = []
    for t in thresholds:
        scores.append(np.mean(distances < t))
    return np.mean(scores)
