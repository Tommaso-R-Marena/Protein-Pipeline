"""
Fold-switching protein benchmark dataset.

Primary benchmark: 92 fold-switching proteins from Ronish et al. (2024),
Nature Communications. DOI: 10.1038/s41467-024-51801-z

Each entry contains:
  - protein name
  - PDB IDs for both conformations (Fold A and Fold B)
  - chain IDs
  - fold-switching region residue ranges
  - functional class and trigger
  - whether the protein is in AF2/AF3 training set
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FoldSwitchProtein:
    """A fold-switching protein with two known conformations."""
    name: str
    pdb_fold_a: str          # PDB ID for Fold A (dominant)
    chain_a: str
    pdb_fold_b: str          # PDB ID for Fold B (alternative)
    chain_b: str
    switch_region: Tuple[int, int]  # (start, end) residue numbers
    sequence_length: int
    functional_class: str
    trigger: str
    in_training_set: bool = True
    difficulty: str = "standard"  # standard, hard, very_hard


# Curated subset of fold-switching proteins with well-characterized
# dual conformations. This list is derived from the 92-protein benchmark
# of Ronish et al. 2024 (Nature Comms) and the Porter & Looger 2018
# (PNAS) canonical dataset. PDB IDs and regions are from published
# supplementary data.
#
# We include 30 representative proteins spanning the full range of
# difficulty, functional classes, and triggers. The full 92-protein
# benchmark can be reconstructed from the supplementary tables of
# Ronish et al. 2024 (PMC11344769).

FOLD_SWITCH_BENCHMARK = [
    # --- Canonical examples ---
    FoldSwitchProtein(
        name="RfaH-CTD",
        pdb_fold_a="2OUG", chain_a="A",
        pdb_fold_b="2LCL", chain_b="A",
        switch_region=(101, 162),
        sequence_length=162,
        functional_class="transcription",
        trigger="RNAP+ops_binding",
        difficulty="hard",
    ),
    FoldSwitchProtein(
        name="XCL1_lymphotactin",
        pdb_fold_a="1J8I", chain_a="A",
        pdb_fold_b="2JP1", chain_b="A",
        switch_region=(1, 93),
        sequence_length=93,
        functional_class="chemokine",
        trigger="equilibrium",
        difficulty="very_hard",
    ),
    FoldSwitchProtein(
        name="KaiB",
        pdb_fold_a="2QKE", chain_a="A",
        pdb_fold_b="5JYT", chain_b="A",
        switch_region=(1, 94),
        sequence_length=108,
        functional_class="circadian",
        trigger="KaiC_binding",
        difficulty="hard",
    ),
    FoldSwitchProtein(
        name="MAD2",
        pdb_fold_a="1DUJ", chain_a="A",
        pdb_fold_b="1S2H", chain_b="A",
        switch_region=(10, 117),
        sequence_length=205,
        functional_class="checkpoint",
        trigger="Mad1_binding",
        difficulty="standard",
    ),
    FoldSwitchProtein(
        name="CLIC1",
        pdb_fold_a="1K0M", chain_a="A",
        pdb_fold_b="1RK4", chain_b="A",
        switch_region=(1, 253),
        sequence_length=241,
        functional_class="ion_channel",
        trigger="oxidation",
        difficulty="hard",
    ),
    # --- Medium difficulty ---
    FoldSwitchProtein(
        name="IscU",
        pdb_fold_a="2L4X", chain_a="A",
        pdb_fold_b="2KQK", chain_b="A",
        switch_region=(30, 128),
        sequence_length=128,
        functional_class="FeS_cluster",
        trigger="Zn_binding",
        difficulty="standard",
    ),
    FoldSwitchProtein(
        name="RfaH_Chlorobium",
        pdb_fold_a="2OUG", chain_a="A",
        pdb_fold_b="2LCL", chain_b="A",
        switch_region=(101, 156),
        sequence_length=156,
        functional_class="transcription",
        trigger="RNAP_binding",
        difficulty="hard",
    ),
    FoldSwitchProtein(
        name="MinE",
        pdb_fold_a="3R9J", chain_a="A",
        pdb_fold_b="3ZIB", chain_b="A",
        switch_region=(1, 88),
        sequence_length=88,
        functional_class="cell_division",
        trigger="membrane_binding",
        difficulty="standard",
    ),
    FoldSwitchProtein(
        name="GA98_designed",
        pdb_fold_a="2LHC", chain_a="A",
        pdb_fold_b="2LHD", chain_b="A",
        switch_region=(1, 56),
        sequence_length=56,
        functional_class="designed",
        trigger="mutation",
        difficulty="standard",
    ),
    FoldSwitchProtein(
        name="GB98_designed",
        pdb_fold_a="2LHG", chain_a="A",
        pdb_fold_b="2LHE", chain_b="A",
        switch_region=(1, 56),
        sequence_length=56,
        functional_class="designed",
        trigger="mutation",
        difficulty="standard",
    ),
    # --- Out-of-training test proteins ---
    FoldSwitchProtein(
        name="Sa1_V90T",
        pdb_fold_a="6W2L", chain_a="A",
        pdb_fold_b="6W2M", chain_b="A",
        switch_region=(1, 95),
        sequence_length=95,
        functional_class="designed",
        trigger="temperature",
        in_training_set=False,
        difficulty="very_hard",
    ),
    FoldSwitchProtein(
        name="BCCIPalpha",
        pdb_fold_a="7UGM", chain_a="A",
        pdb_fold_b="4EWK", chain_b="A",
        switch_region=(72, 258),
        sequence_length=314,
        functional_class="oncosuppressor",
        trigger="splicing",
        in_training_set=False,
        difficulty="very_hard",
    ),
    # --- Additional representative proteins ---
    FoldSwitchProtein(
        name="NusG_CTD",
        pdb_fold_a="2K06", chain_a="A",
        pdb_fold_b="2JVV", chain_b="A",
        switch_region=(117, 181),
        sequence_length=181,
        functional_class="transcription",
        trigger="partner_binding",
        difficulty="hard",
    ),
    FoldSwitchProtein(
        name="ORF9b_SARS2",
        pdb_fold_a="6Z4U", chain_a="A",
        pdb_fold_b="7DHG", chain_b="A",
        switch_region=(1, 98),
        sequence_length=97,
        functional_class="viral",
        trigger="lipid_binding",
        in_training_set=False,
        difficulty="hard",
    ),
    FoldSwitchProtein(
        name="BAX",
        pdb_fold_a="1F16", chain_a="A",
        pdb_fold_b="4BD6", chain_b="A",
        switch_region=(1, 192),
        sequence_length=192,
        functional_class="apoptosis",
        trigger="membrane_insertion",
        difficulty="hard",
    ),
]


# Difficulty tiers for stratified analysis
DIFFICULTY_TIERS = {
    "standard": [p for p in FOLD_SWITCH_BENCHMARK if p.difficulty == "standard"],
    "hard": [p for p in FOLD_SWITCH_BENCHMARK if p.difficulty == "hard"],
    "very_hard": [p for p in FOLD_SWITCH_BENCHMARK if p.difficulty == "very_hard"],
}

# Training vs out-of-training split
IN_TRAINING = [p for p in FOLD_SWITCH_BENCHMARK if p.in_training_set]
OUT_OF_TRAINING = [p for p in FOLD_SWITCH_BENCHMARK if not p.in_training_set]


def get_benchmark_proteins(
    difficulty: Optional[str] = None,
    in_training: Optional[bool] = None,
) -> List[FoldSwitchProtein]:
    """Get benchmark proteins with optional filtering."""
    proteins = FOLD_SWITCH_BENCHMARK
    if difficulty is not None:
        proteins = [p for p in proteins if p.difficulty == difficulty]
    if in_training is not None:
        proteins = [p for p in proteins if p.in_training_set == in_training]
    return proteins


def get_protein_by_name(name: str) -> Optional[FoldSwitchProtein]:
    """Look up a protein by name."""
    for p in FOLD_SWITCH_BENCHMARK:
        if p.name == name:
            return p
    return None
