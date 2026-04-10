"""
Benchmark validation set loader for heterochiral complex evaluation.

Based on the 13-system test set from:
  Childs et al. (2025) "Has AlphaFold 3 Solved the Protein Folding Problem for D-Peptides?"
  bioRxiv 2025.03.14.643307

PDB IDs for crystal structure retrospectives:
  - 7YH8: DP19:L-19437  (19-residue D-peptide, L-protein target)
  - 5N8T: DP9:Streptavidin (9-residue D-peptide)
  - 3IWY: DP12:MDM2       (12-residue D-peptide)

Apo D-protein:
  - 1SHG: L-SH3 domain (reflected to D-space for apo D-protein test)

These PDB IDs are fetched from RCSB on first use and cached locally.
"""

CRYSTAL_BENCHMARK_PDB_IDS = {
    "DP19_L19437":      "7YH8",
    "DP9_Streptavidin": "5N8T",
    "DP12_MDM2":        "3IWY",
    "Apo_D_SH3":        "1SHG",   # reflected
}

SYNTHETIC_BENCHMARK_TARGETS = [
    {"name": "DP_Ubiquitin_decoy", "pdb": "6NXL"},
    {"name": "DP_GB1_decoy",       "pdb": "2J52"},
]

def get_benchmark_systems() -> list[dict]:
    """Return metadata for all benchmark systems."""
    systems = []
    for name, pdb_id in CRYSTAL_BENCHMARK_PDB_IDS.items():
        systems.append({
            "name":    name,
            "pdb_id":  pdb_id,
            "type":    "crystal",
            "is_apo":  name.startswith("Apo"),
            "reflect": name.startswith("Apo"),
        })
    for sys in SYNTHETIC_BENCHMARK_TARGETS:
        systems.append({**sys, "type": "synthetic", "is_apo": False, "reflect": False})
    return systems
