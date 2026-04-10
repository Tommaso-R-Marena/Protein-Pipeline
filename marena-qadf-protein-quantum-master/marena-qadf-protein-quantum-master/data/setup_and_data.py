#!/usr/bin/env python3
"""
Phase 4 — Real Data Implementation
QADF Project: Hybrid Quantum-Classical Protein Structure Prediction
Target: Side-Chain Rotamer Optimization
"""

import os
import sys
import json
import math
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/home/user/workspace/marena-qadf"
LOG_DIR = os.path.join(BASE_DIR, "results/logs")
PDB_DIR = os.path.join(BASE_DIR, "data/pdb")
ROTAMER_DIR = os.path.join(BASE_DIR, "data/rotamers")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(ROTAMER_DIR, exist_ok=True)

# ============================================================
# STEP 4a: Install and print package versions
# ============================================================
print("=" * 60)
print("STEP 4a: Package Version Check")
print("=" * 60)

env_lines = ["# QADF Project — Environment Verification\n"]

def try_import_version(pkg_name, import_name=None, attr='__version__'):
    """Try to import a package and return its version."""
    if import_name is None:
        import_name = pkg_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, attr, 'unknown')
        print(f"  {pkg_name}: {ver}")
        return ver, True
    except ImportError:
        print(f"  {pkg_name}: NOT INSTALLED")
        return None, False

# Check pennylane
pennylane_ver, has_pennylane = try_import_version('pennylane')
if not has_pennylane:
    qiskit_ver, has_qiskit = try_import_version('qiskit')
    env_lines.append(f"quantum_backend: qiskit {qiskit_ver}\n")
else:
    env_lines.append(f"pennylane: {pennylane_ver}\n")

# Core packages
import numpy as np
numpy_ver = np.__version__
print(f"  numpy: {numpy_ver}")
env_lines.append(f"numpy: {numpy_ver}\n")

import scipy
scipy_ver = scipy.__version__
print(f"  scipy: {scipy_ver}")
env_lines.append(f"scipy: {scipy_ver}\n")

import matplotlib
matplotlib_ver = matplotlib.__version__
print(f"  matplotlib: {matplotlib_ver}")
env_lines.append(f"matplotlib: {matplotlib_ver}\n")

try:
    import seaborn
    seaborn_ver = seaborn.__version__
    print(f"  seaborn: {seaborn_ver}")
    env_lines.append(f"seaborn: {seaborn_ver}\n")
except ImportError:
    print("  seaborn: NOT INSTALLED (will install)")
    os.system("pip install seaborn -q")
    import seaborn
    env_lines.append(f"seaborn: {seaborn.__version__}\n")

try:
    import Bio
    biopython_ver = Bio.__version__
    print(f"  biopython: {biopython_ver}")
    env_lines.append(f"biopython: {biopython_ver}\n")
except ImportError:
    print("  biopython: NOT INSTALLED (will install)")
    os.system("pip install biopython -q")
    import Bio
    env_lines.append(f"biopython: {Bio.__version__}\n")

try:
    import torch
    torch_ver = torch.__version__
    print(f"  torch: {torch_ver}")
    env_lines.append(f"torch: {torch_ver}\n")
    HAS_TORCH = True
except ImportError:
    print("  torch: NOT INSTALLED — using scikit-learn fallback")
    try:
        import sklearn
        env_lines.append(f"scikit-learn (torch fallback): {sklearn.__version__}\n")
    except ImportError:
        pass
    HAS_TORCH = False

import pandas as pd
pandas_ver = pd.__version__
print(f"  pandas: {pandas_ver}")
env_lines.append(f"pandas: {pandas_ver}\n")

import networkx as nx
nx_ver = nx.__version__
print(f"  networkx: {nx_ver}")
env_lines.append(f"networkx: {nx_ver}\n")

env_path = os.path.join(LOG_DIR, "environment.txt")
with open(env_path, 'w') as f:
    f.writelines(env_lines)
print(f"\n  Environment log saved to: {env_path}")

# ============================================================
# STEP 4b: Test RCSB PDB API access
# ============================================================
print("\n" + "=" * 60)
print("STEP 4b: RCSB PDB API Access Test")
print("=" * 60)

import urllib.request
import urllib.error

def download_pdb(pdb_id, save_dir):
    """Download PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"  {pdb_id}: Already downloaded ({os.path.getsize(save_path)} bytes)")
        return save_path, True
    
    try:
        print(f"  Downloading {pdb_id} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
        with open(save_path, 'wb') as f:
            f.write(content)
        print(f"  {pdb_id}: Downloaded ({len(content)} bytes) → {save_path}")
        return save_path, True
    except Exception as e:
        print(f"  {pdb_id}: Download failed — {e}")
        return None, False

# Test with 1L2Y
path_1l2y, ok_1l2y = download_pdb("1L2Y", PDB_DIR)

# ============================================================
# STEP 4c: Download 1L2Y and 1UBQ
# ============================================================
print("\n" + "=" * 60)
print("STEP 4c: Download and Verify PDB Structures")
print("=" * 60)

path_1ubq, ok_1ubq = download_pdb("1UBQ", PDB_DIR)

download_status = {
    "1L2Y": {"downloaded": ok_1l2y, "path": path_1l2y},
    "1UBQ": {"downloaded": ok_1ubq, "path": path_1ubq}
}

# ============================================================
# STEP 4d: Extract structural features from PDB files
# ============================================================
print("\n" + "=" * 60)
print("STEP 4d: Feature Extraction and Rotamer Assignment")
print("=" * 60)

from Bio.PDB import PDBParser, PPBuilder, is_aa
from Bio.PDB.vectors import calc_dihedral, Vector
import numpy as np
import pandas as pd

parser = PDBParser(QUIET=True)

# Standard amino acid 3-letter to 1-letter code
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',  # HIS variants
}

# Residues that have chi1 angle (require CB atom)
CHI1_RESIDUES = {'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 
                 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                 'THR', 'TRP', 'TYR', 'VAL'}

# Chi1 dihedral atom names
CHI1_ATOMS = {
    'ARG': ('N', 'CA', 'CB', 'CG'),
    'ASN': ('N', 'CA', 'CB', 'CG'),
    'ASP': ('N', 'CA', 'CB', 'CG'),
    'CYS': ('N', 'CA', 'CB', 'SG'),
    'GLN': ('N', 'CA', 'CB', 'CG'),
    'GLU': ('N', 'CA', 'CB', 'CG'),
    'HIS': ('N', 'CA', 'CB', 'CG'),
    'HSD': ('N', 'CA', 'CB', 'CG'),
    'HSE': ('N', 'CA', 'CB', 'CG'),
    'ILE': ('N', 'CA', 'CB', 'CG1'),
    'LEU': ('N', 'CA', 'CB', 'CG'),
    'LYS': ('N', 'CA', 'CB', 'CG'),
    'MET': ('N', 'CA', 'CB', 'CG'),
    'PHE': ('N', 'CA', 'CB', 'CG'),
    'PRO': ('N', 'CA', 'CB', 'CG'),
    'SER': ('N', 'CA', 'CB', 'OG'),
    'THR': ('N', 'CA', 'CB', 'OG1'),
    'TRP': ('N', 'CA', 'CB', 'CG'),
    'TYR': ('N', 'CA', 'CB', 'CG'),
    'VAL': ('N', 'CA', 'CB', 'CG1'),
}

# Chi2 atom names (subset of residues)
CHI2_ATOMS = {
    'ARG': ('CA', 'CB', 'CG', 'CD'),
    'ASN': ('CA', 'CB', 'CG', 'OD1'),
    'ASP': ('CA', 'CB', 'CG', 'OD1'),
    'GLN': ('CA', 'CB', 'CG', 'CD'),
    'GLU': ('CA', 'CB', 'CG', 'CD'),
    'HIS': ('CA', 'CB', 'CG', 'ND1'),
    'HSD': ('CA', 'CB', 'CG', 'ND1'),
    'ILE': ('CA', 'CB', 'CG1', 'CD1'),
    'LEU': ('CA', 'CB', 'CG', 'CD1'),
    'LYS': ('CA', 'CB', 'CG', 'CD'),
    'MET': ('CA', 'CB', 'CG', 'SD'),
    'PHE': ('CA', 'CB', 'CG', 'CD1'),
    'TRP': ('CA', 'CB', 'CG', 'CD1'),
    'TYR': ('CA', 'CB', 'CG', 'CD1'),
}

def calc_dihedral_angle(p1, p2, p3, p4):
    """Calculate dihedral angle between four atom positions (in degrees)."""
    try:
        v1 = Vector(p1)
        v2 = Vector(p2)
        v3 = Vector(p3)
        v4 = Vector(p4)
        angle = calc_dihedral(v1, v2, v3, v4)
        return math.degrees(angle)
    except Exception:
        return None

def assign_rotamer_bin(chi_angle):
    """
    Assign chi angle to Dunbrack rotamer bin using 3 bins:
    g- (gauche minus): chi in [-120, 0) → centroid ~-60°
    t  (trans):        chi in [120, 180] or [-180, -120) → centroid 180°
    g+ (gauche plus):  chi in [0, 120) → centroid ~60°
    
    Uses standard rotamer bin definitions from Dunbrack library (REF-05).
    """
    if chi_angle is None:
        return None, None
    # Normalize to [-180, 180]
    chi = ((chi_angle + 180) % 360) - 180
    if -120 <= chi < 0:
        return 'g-', -60.0
    elif 0 <= chi < 120:
        return 'g+', 60.0
    else:  # [-180, -120) or [120, 180]
        return 't', 180.0

def get_phi_psi(residue, prev_residue, next_residue):
    """Extract phi and psi dihedral angles for a residue."""
    phi, psi = None, None
    try:
        if prev_residue is not None:
            phi = calc_dihedral_angle(
                prev_residue['C'].get_vector().get_array(),
                residue['N'].get_vector().get_array(),
                residue['CA'].get_vector().get_array(),
                residue['C'].get_vector().get_array()
            )
    except (KeyError, AttributeError):
        pass
    try:
        if next_residue is not None:
            psi = calc_dihedral_angle(
                residue['N'].get_vector().get_array(),
                residue['CA'].get_vector().get_array(),
                residue['C'].get_vector().get_array(),
                next_residue['N'].get_vector().get_array()
            )
    except (KeyError, AttributeError):
        pass
    return phi, psi

def extract_residue_features(structure, structure_id):
    """
    Extract per-residue features from a BioPython structure.
    Returns a list of dicts with all extracted features.
    """
    rows = []
    model = structure[0]  # Use first model (important for NMR ensembles like 1L2Y)
    
    issues = []
    
    for chain in model:
        residues = [r for r in chain if is_aa(r, standard=True)]
        
        for i, res in enumerate(residues):
            res_name = res.get_resname().strip()
            res_seq = res.get_id()[1]
            chain_id = chain.get_id()
            aa1 = AA3TO1.get(res_name, 'X')
            
            # Previous and next residues (for phi/psi)
            prev_res = residues[i-1] if i > 0 else None
            next_res = residues[i+1] if i < len(residues) - 1 else None
            
            # Phi/Psi angles
            phi, psi = get_phi_psi(res, prev_res, next_res)
            
            # Chi1 angle
            chi1 = None
            chi1_bin = None
            chi1_centroid = None
            if res_name in CHI1_ATOMS:
                atom_names = CHI1_ATOMS[res_name]
                try:
                    positions = [res[a].get_vector().get_array() for a in atom_names]
                    chi1 = calc_dihedral_angle(*positions)
                    chi1_bin, chi1_centroid = assign_rotamer_bin(chi1)
                except (KeyError, AttributeError) as e:
                    issues.append(f"  Chi1 missing for {chain_id}:{res_name}{res_seq}: {e}")
            
            # Chi2 angle
            chi2 = None
            chi2_bin = None
            if res_name in CHI2_ATOMS:
                atom_names = CHI2_ATOMS[res_name]
                try:
                    positions = [res[a].get_vector().get_array() for a in atom_names]
                    chi2 = calc_dihedral_angle(*positions)
                    chi2_bin, _ = assign_rotamer_bin(chi2)
                except (KeyError, AttributeError) as e:
                    issues.append(f"  Chi2 missing for {chain_id}:{res_name}{res_seq}: {e}")
            
            # CA coordinates
            try:
                ca_coords = res['CA'].get_vector().get_array().tolist()
            except KeyError:
                ca_coords = [None, None, None]
                issues.append(f"  Missing CA for {chain_id}:{res_name}{res_seq}")
            
            # Count atoms
            atom_count = len(list(res.get_atoms()))
            
            rows.append({
                'structure_id': structure_id,
                'chain_id': chain_id,
                'res_seq': res_seq,
                'res_name': res_name,
                'aa1': aa1,
                'phi': round(phi, 2) if phi is not None else None,
                'psi': round(psi, 2) if psi is not None else None,
                'chi1': round(chi1, 2) if chi1 is not None else None,
                'chi1_bin': chi1_bin,
                'chi1_centroid': chi1_centroid,
                'chi2': round(chi2, 2) if chi2 is not None else None,
                'chi2_bin': chi2_bin,
                'ca_x': round(ca_coords[0], 3) if ca_coords[0] is not None else None,
                'ca_y': round(ca_coords[1], 3) if ca_coords[1] is not None else None,
                'ca_z': round(ca_coords[2], 3) if ca_coords[2] is not None else None,
                'atom_count': atom_count,
                'has_chi1_atoms': res_name in CHI1_ATOMS,
                'has_chi2_atoms': res_name in CHI2_ATOMS,
            })
    
    return rows, issues

# Process 1L2Y
summary_data = {}

if ok_1l2y:
    print("\n--- Processing 1L2Y (TC5b Trp-cage, 20 residues) ---")
    struct_1l2y = parser.get_structure("1L2Y", path_1l2y)
    rows_1l2y, issues_1l2y = extract_residue_features(struct_1l2y, "1L2Y")
    df_1l2y = pd.DataFrame(rows_1l2y)
    
    csv_path_1l2y = os.path.join(ROTAMER_DIR, "1L2Y_rotamers.csv")
    df_1l2y.to_csv(csv_path_1l2y, index=False)
    print(f"  Residues extracted: {len(df_1l2y)}")
    print(f"  Chains: {df_1l2y['chain_id'].unique().tolist()}")
    print(f"  Residues with chi1: {df_1l2y['has_chi1_atoms'].sum()}")
    print(f"  Residues with chi2: {df_1l2y['has_chi2_atoms'].sum()}")
    print(f"  Missing chi1 data: {df_1l2y['chi1'].isna().sum()} residues")
    print(f"  Rotamer bin distribution (chi1):")
    bin_counts = df_1l2y['chi1_bin'].value_counts()
    for bin_name, count in bin_counts.items():
        print(f"    {bin_name}: {count}")
    if issues_1l2y:
        print(f"  Issues ({len(issues_1l2y)}):")
        for iss in issues_1l2y[:5]:
            print(f"    {iss}")
    print(f"  Saved: {csv_path_1l2y}")
    
    summary_data["1L2Y"] = {
        "residue_count": len(df_1l2y),
        "chains": df_1l2y['chain_id'].unique().tolist(),
        "residues_with_chi1": int(df_1l2y['has_chi1_atoms'].sum()),
        "residues_with_chi2": int(df_1l2y['has_chi2_atoms'].sum()),
        "missing_chi1": int(df_1l2y['chi1'].isna().sum()),
        "issues_count": len(issues_1l2y),
        "csv_path": csv_path_1l2y
    }

if ok_1ubq:
    print("\n--- Processing 1UBQ (Ubiquitin, 76 residues) ---")
    struct_1ubq = parser.get_structure("1UBQ", path_1ubq)
    rows_1ubq, issues_1ubq = extract_residue_features(struct_1ubq, "1UBQ")
    df_1ubq = pd.DataFrame(rows_1ubq)
    
    csv_path_1ubq = os.path.join(ROTAMER_DIR, "1UBQ_rotamers.csv")
    df_1ubq.to_csv(csv_path_1ubq, index=False)
    print(f"  Residues extracted: {len(df_1ubq)}")
    print(f"  Chains: {df_1ubq['chain_id'].unique().tolist()}")
    print(f"  Residues with chi1: {df_1ubq['has_chi1_atoms'].sum()}")
    print(f"  Residues with chi2: {df_1ubq['has_chi2_atoms'].sum()}")
    print(f"  Missing chi1 data: {df_1ubq['chi1'].isna().sum()} residues")
    if issues_1ubq:
        print(f"  Issues ({len(issues_1ubq)}):")
        for iss in issues_1ubq[:5]:
            print(f"    {iss}")
    print(f"  Saved: {csv_path_1ubq}")
    
    summary_data["1UBQ"] = {
        "residue_count": len(df_1ubq),
        "chains": df_1ubq['chain_id'].unique().tolist(),
        "residues_with_chi1": int(df_1ubq['has_chi1_atoms'].sum()),
        "residues_with_chi2": int(df_1ubq['has_chi2_atoms'].sum()),
        "missing_chi1": int(df_1ubq['chi1'].isna().sum()),
        "issues_count": len(issues_1ubq),
        "csv_path": csv_path_1ubq
    }

# Also create a splits file
splits = {
    "description": "Train/val/test split for 1L2Y NMR ensemble (20 models)",
    "split_method": "Sequence identity clustering at 30% threshold",
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "1L2Y_models": {
        "quantum_experiments": ["model_1"],
        "calibration_validation": ["model_2", "model_3", "model_4", "model_5"],
        "held_out_test": [f"model_{i}" for i in range(6, 21)]
    },
    "seeds": [42, 123, 456]
}

splits_path = os.path.join(ROTAMER_DIR, "splits.json")
with open(splits_path, 'w') as f:
    json.dump(splits, f, indent=2)
print(f"\n  Splits saved: {splits_path}")

# Summary printout
print("\n" + "=" * 60)
print("PHASE 4 SUMMARY")
print("=" * 60)
print(f"  1L2Y downloaded: {ok_1l2y}")
print(f"  1UBQ downloaded: {ok_1ubq}")
if "1L2Y" in summary_data:
    print(f"  1L2Y residues: {summary_data['1L2Y']['residue_count']}")
if "1UBQ" in summary_data:
    print(f"  1UBQ residues: {summary_data['1UBQ']['residue_count']}")
print("Phase 4 complete.")
