"""Tests for mirror-image coordinate transformation."""
import numpy as np
import torch

from chiralboltz.augmentation.mirror import (
    reflect_coords_numpy,
    reflect_coords_tensor,
    flip_chirality_flags,
)


def test_reflect_coords_numpy_negates_x():
    coords = np.array([[1.0, 2.0, 3.0], [-1.0, 4.0, -5.0]])
    out = reflect_coords_numpy(coords)
    np.testing.assert_allclose(out[:, 0], -coords[:, 0])
    np.testing.assert_allclose(out[:, 1],  coords[:, 1])
    np.testing.assert_allclose(out[:, 2],  coords[:, 2])


def test_reflect_coords_numpy_double_reflection_is_identity():
    coords = np.random.randn(100, 3).astype(np.float32)
    np.testing.assert_allclose(
        reflect_coords_numpy(reflect_coords_numpy(coords)), coords, atol=1e-6
    )


def test_reflect_coords_tensor_differentiable():
    coords = torch.randn(4, 20, 3, requires_grad=True)
    out = reflect_coords_tensor(coords)
    out.sum().backward()
    assert coords.grad is not None, "Gradients should flow through reflect_coords_tensor"
    # x-channel gradient should be -1, y and z should be +1
    assert torch.allclose(coords.grad[..., 0], -torch.ones_like(coords.grad[..., 0]))
    assert torch.allclose(coords.grad[..., 1],  torch.ones_like(coords.grad[..., 1]))


def test_flip_chirality_cw_ccw():
    dt = np.dtype([('name', 'U4'), ('chirality', np.int32)])
    atoms = np.array([('CA', 1), ('N', 0), ('CB', 2), ('C', 1)], dtype=dt)
    flipped = flip_chirality_flags(atoms)
    assert flipped['chirality'][0] == 2   # CW -> CCW
    assert flipped['chirality'][1] == 0   # achiral unchanged
    assert flipped['chirality'][2] == 1   # CCW -> CW
    assert flipped['chirality'][3] == 2   # CW -> CCW


def test_flip_chirality_double_flip_is_identity():
    dt = np.dtype([('name', 'U4'), ('chirality', np.int32)])
    atoms = np.array([('CA', 1), ('CB', 2), ('CG', 0)], dtype=dt)
    assert np.array_equal(
        flip_chirality_flags(flip_chirality_flags(atoms))['chirality'],
        atoms['chirality']
    )
