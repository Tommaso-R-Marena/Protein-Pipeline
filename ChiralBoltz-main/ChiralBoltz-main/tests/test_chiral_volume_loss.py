"""Tests for the differentiable chiral volume loss."""
import torch
from chiralboltz.loss.chiral_volume import signed_chiral_volume, chiral_volume_loss


def _make_l_center():
    """Return coords for an idealized L-alanine alpha carbon chiral center."""
    # Standard L-amino acid tetrahedral geometry: positive signed volume
    # V = (r_c - r_j) · ((r_c - r_k) × (r_c - r_l))
    # With this ordering: V = (-1,0,0) · ((-1,0,0)×(0,-1,0)) -> not right-handed
    # Swap k and l to get positive volume: V = (0,-1,0) cross (0,0,-1) = (1,0,0)
    # dot with (-1,0,0) = -1 still. Use different geometry instead.
    r_c = torch.tensor([0.0,  0.0,  0.0])
    r_j = torch.tensor([0.0,  0.0,  1.0])
    r_k = torch.tensor([0.0,  1.0,  0.0])
    r_l = torch.tensor([1.0,  0.0,  0.0])
    return r_c, r_j, r_k, r_l


def test_signed_chiral_volume_positive_for_L():
    r_c, r_j, r_k, r_l = _make_l_center()
    V = signed_chiral_volume(r_c, r_j, r_k, r_l)
    assert V.item() > 0, "L-center should have positive signed volume"


def test_signed_chiral_volume_negates_under_reflection():
    r_c, r_j, r_k, r_l = _make_l_center()
    V_L = signed_chiral_volume(r_c, r_j, r_k, r_l)
    # Reflect x -> -x
    def reflect(t): return t * torch.tensor([-1.0, 1.0, 1.0])
    V_D = signed_chiral_volume(reflect(r_c), reflect(r_j), reflect(r_k), reflect(r_l))
    assert torch.allclose(V_D, -V_L, atol=1e-5), "Reflection should negate chiral volume"


def test_chiral_volume_loss_zero_for_correct_chirality():
    """Loss should be ~0 when predicted chirality matches reference."""
    B, N = 2, 10
    coords = torch.randn(B, N, 3)
    # chiral_centers: center=0, j=1, k=2, l=3
    centers = torch.zeros(B, 1, 4, dtype=torch.long)
    centers[:, 0, :] = torch.tensor([0, 1, 2, 3])
    mask = torch.ones(B, 1, dtype=torch.bool)

    # pred == true => loss should be very small
    loss = chiral_volume_loss(
        pred_coords=coords,
        true_coords=coords,
        chiral_centers=centers,
        chiral_mask=mask,
        mode="sign",
    )
    assert loss.item() < 0.2, f"Loss for identical coords should be small, got {loss.item()}"


def test_chiral_volume_loss_large_for_inverted_chirality():
    """Loss should be large when chirality is inverted."""
    B, N = 2, 10
    # Use a deterministic chiral center: known positive volume
    coords_true = torch.zeros(B, N, 3)
    coords_true[:, 0] = torch.tensor([0.0, 0.0, 0.0])  # center
    coords_true[:, 1] = torch.tensor([1.0, 0.0, 0.0])
    coords_true[:, 2] = torch.tensor([0.0, 1.0, 0.0])
    coords_true[:, 3] = torch.tensor([0.0, 0.0, 1.0])

    # Inverted: reflect x -> -x for center and all neighbors
    coords_pred = coords_true.clone()
    coords_pred[:, :4, 0] = -coords_pred[:, :4, 0]

    centers = torch.zeros(B, 1, 4, dtype=torch.long)
    centers[:, 0, :] = torch.tensor([0, 1, 2, 3])
    mask = torch.ones(B, 1, dtype=torch.bool)

    loss = chiral_volume_loss(
        pred_coords=coords_pred,
        true_coords=coords_true,
        chiral_centers=centers,
        chiral_mask=mask,
        mode="sign",
        sign_margin=0.1,
    )
    assert loss.item() > 0.5, f"Inverted chirality should give large loss, got {loss.item()}"


def test_chiral_volume_loss_differentiable():
    B, N = 2, 10
    coords_pred = torch.randn(B, N, 3, requires_grad=True)
    coords_true = torch.randn(B, N, 3)
    centers = torch.zeros(B, 1, 4, dtype=torch.long)
    centers[:, 0, :] = torch.tensor([0, 1, 2, 3])
    mask = torch.ones(B, 1, dtype=torch.bool)

    loss = chiral_volume_loss(coords_pred, coords_true, centers, mask, mode="combined")
    loss.backward()
    assert coords_pred.grad is not None
    assert not torch.isnan(coords_pred.grad).any()
