"""
MirrorAugmentedDataset: wraps a Boltz-2 training dataset and applies
mirror-image augmentation with configurable probability.

Design:
  - At each __getitem__ call, flip a Bernoulli(p_mirror) coin.
  - If heads: apply mirror_structure_v2 to the structure before featurization.
  - All MSA features are unaffected (sequence is chirality-invariant).
  - The resulting batch is labelled with an 'is_mirrored' boolean flag
    for debugging and analysis.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from chiralboltz.augmentation.mirror import mirror_structure_v2


class MirrorAugmentedDataset(Dataset):
    """
    Wraps any Boltz-2-compatible dataset and applies random mirror-image augmentation.

    Parameters
    ----------
    base_dataset : Dataset
        A Boltz-2 training Dataset instance (from boltz.data.module.trainingv2).
    p_mirror : float
        Probability of applying the mirror augmentation to each sample.
        Default 0.5 gives equal L and D training examples.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, base_dataset: Dataset, p_mirror: float = 0.5, seed: int = 42):
        self.base_dataset = base_dataset
        self.p_mirror = p_mirror
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx]
        should_mirror = self.rng.random() < self.p_mirror

        if should_mirror and 'structure' in sample:
            sample['structure'] = mirror_structure_v2(sample['structure'])

        sample['is_mirrored'] = torch.tensor(should_mirror, dtype=torch.bool)
        return sample
