#!/usr/bin/env python
"""
Prepare mirror-image dataset from Boltz-2 NPZ structure files.

Usage:
    python scripts/prepare_mirror_dataset.py \
        --source_dir /path/to/rcsb_processed_targets/structures \
        --output_dir /path/to/mirror_structures \
        --n_workers 8
"""
import argparse
from pathlib import Path
from chiralboltz.data.pdb_mirror import mirror_directory


def main():
    parser = argparse.ArgumentParser(description="Mirror-image dataset preparation")
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_workers",  type=int,  default=4)
    args = parser.parse_args()
    mirror_directory(args.source_dir, args.output_dir, args.n_workers)


if __name__ == "__main__":
    main()
