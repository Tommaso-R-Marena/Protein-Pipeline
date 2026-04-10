"""
Dataset manifests for standard benchmarking within QuantumProteinOS.
"""

AUTOINHIBITED_22 = [
    "ABL1", "SRC", "HCK", "EGFR", "STAT3", "PIK3CA", "WAS", "BRAF", "AK1",
    "CDK2", "ERK2", "GSK3B", "MAPK14", "PAK1", "PKA", "PKB", "PKC", "SYK",
    "TEC", "TYK2", "ZAP70", "LCK"
]

FOLD_SWITCH_92 = [
    # Demo/Mock entries (5 demo proteins from Ronish et al.)
    ("1J4N", "1J4O"), # Example pair
    ("2L3Y", "2L3Z"),
    ("3O4A", "3O4B"),
    ("4P5C", "4P5D"),
    ("5Q6E", "5Q6F")
    # ... placeholder for the rest of 92
]

AUTOINHIBITED_14 = [
    "ABL1", "AK1", "BRAF", "CBS", "EGFR", "FGFR1", "FYN", "HCK", 
    "JAK1", "KIT", "LCK", "PTPN6", "SRC", "STAT3"
]

QADF_SUBPROBLEMS = [
    "Rotamer Packing",
    "Loop Modeling",
    "Molecular Docking",
    "Protein Folding (Lattice)",
    "Protein Folding (All-Atom)",
    "Sequence Alignment",
    "Drug Discovery (Virtual Screening)",
    "Protein Design"
]

# QADF v2 39-protein dataset with splits
QADF_V2_39 = {
    "train": [
        "1L2Y", "1UBQ", "1CRN", "1BPI", "2I9M", "1FSD", "1PTQ", "1VII",
        "2F4K", "1A3A", "1ENH", "1PRW", "1WQ5", "2JOF", "2A3D", "1DFN",
        "1ZDD", "1G6P", "1CBP", "1E0L", "1H1V"
    ],
    "val": [
        "1KNL", "2KJU", "1BBL", "2A11", "1ROO", "2JWS", "1T8K", "2KQM", "1F94"
    ],
    "test": [
        "1E0G", "1A11", "2A0B", "1J2G", "1ZJI", "1HZY", "1F4I", "2M7M", "2J3K"
    ]
}
