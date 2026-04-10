"""Tests for MirrorAugmentedDataset."""
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset

from chiralboltz.augmentation.dataset import MirrorAugmentedDataset


class _FakeDataset(Dataset):
    """Minimal fake dataset that returns structure-containing dicts."""
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'structure': MagicMock(
                coords=np.random.randn(50, 3).astype(np.float32),
                ensemble=np.random.randn(5, 50, 3).astype(np.float32),
                atoms=np.zeros(50, dtype=[('name','U4'),('chirality', np.int32)]),
                bonds=np.zeros((10,2), dtype=np.int32),
                residues=np.zeros(5, dtype=np.int32),
                chains=np.zeros(2, dtype=np.int32),
                interfaces=np.zeros((2,2), dtype=np.int32),
                mask=np.ones(50, dtype=bool),
                pocket=None,
            ),
            'id': idx,
        }


@patch('chiralboltz.augmentation.dataset.mirror_structure_v2', side_effect=lambda s: s)
def test_mirror_augmented_dataset_returns_is_mirrored_flag(mock_mirror):
    ds = MirrorAugmentedDataset(_FakeDataset(10), p_mirror=0.5, seed=0)
    sample = ds[0]
    assert 'is_mirrored' in sample
    assert isinstance(sample['is_mirrored'], torch.Tensor)


def test_mirror_augmented_dataset_p_mirror_zero_never_mirrors():
    ds = MirrorAugmentedDataset(_FakeDataset(20), p_mirror=0.0, seed=0)
    for i in range(20):
        assert not ds[i]['is_mirrored'].item()


@patch('chiralboltz.augmentation.dataset.mirror_structure_v2', side_effect=lambda s: s)
def test_mirror_augmented_dataset_p_mirror_one_always_mirrors(mock_mirror):
    ds = MirrorAugmentedDataset(_FakeDataset(20), p_mirror=1.0, seed=0)
    for i in range(20):
        assert ds[i]['is_mirrored'].item()


def test_mirror_augmented_dataset_length():
    base = _FakeDataset(42)
    ds   = MirrorAugmentedDataset(base, p_mirror=0.5)
    assert len(ds) == 42
