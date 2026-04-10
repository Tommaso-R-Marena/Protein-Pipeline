"""
Batch mirror-image preprocessing for PDB NPZ structures.

Reads Boltz-2-format NPZ structure files from a source directory,
applies mirror_structure_v2, and writes to an output directory.
The resulting files are drop-in replacements in any Boltz-2 training config.

Usage:
    python scripts/prepare_mirror_dataset.py \
        --source_dir data/structures \
        --output_dir data/mirror_structures \
        --n_workers 8

This effectively doubles the training set: every L-protein becomes a D-protein.
"""

import multiprocessing
from pathlib import Path

import numpy as np

from chiralboltz.augmentation.mirror import reflect_coords_numpy, flip_chirality_flags


def mirror_npz(src_path: Path, dst_path: Path) -> None:
    """
    Load a Boltz-2 NPZ structure, apply mirror reflection, save to dst_path.

    Handles both StructureV1 (atoms/bonds/residues/chains/connections/interfaces/mask)
    and StructureV2 (adds coords/ensemble fields).
    """
    data = np.load(src_path, allow_pickle=False)
    out  = dict(data)

    # Reflect coordinate fields if present (StructureV2)
    for coord_field in ('coords', 'ensemble'):
        if coord_field in out:
            out[coord_field] = reflect_coords_numpy(out[coord_field])

    # Flip chirality in atoms structured array
    if 'atoms' in out and out['atoms'].dtype.names and 'chirality' in out['atoms'].dtype.names:
        out['atoms'] = flip_chirality_flags(out['atoms'])

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_path, **out)


def mirror_npz_worker(args: tuple) -> None:
    src, dst = args
    mirror_npz(Path(src), Path(dst))


def mirror_directory(source_dir: Path, output_dir: Path, n_workers: int = 4) -> None:
    """
    Mirror all NPZ structure files in source_dir to output_dir.

    Parameters
    ----------
    source_dir : Path
        Directory containing Boltz-2 NPZ structure files.
    output_dir : Path
        Destination directory (will be created).
    n_workers : int
        Number of parallel worker processes.
    """
    src_files = list(source_dir.rglob("*.npz"))
    tasks = [
        (str(f), str(output_dir / f.relative_to(source_dir)))
        for f in src_files
    ]

    print(f"Mirroring {len(tasks)} structures with {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(mirror_npz_worker, tasks)
    print("Done.")
