"""Tests de las defensas."""

import torch

from adversarial_playground.defenses.smoothing import RandomizedSmoothing
from adversarial_playground.defenses.training import AdvTrainingConfig, adversarial_loss
from adversarial_playground.defenses.transforms import (
    BitDepthReduction,
    GaussianNoise,
    JPEGCompression,
    TotalVarianceMinimization,
)


# ----------------------------------------------------- transformaciones ----
def test_bit_depth_reduces_levels():
    defense = BitDepthReduction(bits=3)
    x = torch.rand(4, 1, 8, 8)
    out = defense(x)
    assert torch.unique(out).numel() <= 2**3
    assert out.min() >= 0 and out.max() <= 1


def test_bit_depth_invalid_bits_raises():
    import pytest

    with pytest.raises(ValueError):
        BitDepthReduction(bits=9)


def test_jpeg_roundtrip_preserves_shape_and_range():
    defense = JPEGCompression(quality=75)
    x = torch.rand(2, 1, 28, 28)
    out = defense(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_gaussian_noise_clamps():
    defense = GaussianNoise(sigma=5.0)  # ruido absurdo: debe seguir en [0, 1]
    out = defense(torch.full((1, 1, 4, 4), 0.5))
    assert out.min() >= 0 and out.max() <= 1


def test_tvm_runs():
    defense = TotalVarianceMinimization(steps=5)
    x = torch.rand(2, 1, 12, 12)
    out = defense(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    # Suaviza: la variación total de la salida no supera la de la entrada
    def tv(img):
        dh = (img[..., :, 1:] - img[..., :, :-1]).abs().sum()
        dv = (img[..., 1:, :] - img[..., :-1, :]).abs().sum()
        return float(dh + dv)

    assert tv(out) < tv(x)


# ----------------------------------------------- entrenamiento adversarial ----
def test_adversarial_loss_runs_and_backprops():
    from adversarial_playground.models import MnistCNN

    torch.manual_seed(0)
    model = MnistCNN()
    x = torch.rand(8, 1, 28, 28)
    y = model(x).argmax(1)

    cfg = AdvTrainingConfig(eps=0.3, alpha=0.075, steps=3)
    loss, stats = adversarial_loss(model, x, y, cfg)
    loss.backward()

    assert torch.isfinite(loss)
    assert "acc_adv" in stats
    assert any(p.grad is not None for p in model.parameters())


def test_adversarial_loss_mixed_mode():
    from adversarial_playground.models import MnistCNN

    torch.manual_seed(0)
    model = MnistCNN()
    x = torch.rand(8, 1, 28, 28)
    y = model(x).argmax(1)

    cfg = AdvTrainingConfig(eps=0.3, alpha=0.075, steps=2)
    loss, stats = adversarial_loss(model, x, y, cfg, mix_ratio=0.5)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(stats["acc_adv"]))


# ------------------------------------------------- randomized smoothing ----
def test_smoothing_predict_majority_with_tiny_noise(model):
    torch.manual_seed(0)
    x = torch.rand(4, 1, 28, 28)
    smooth = RandomizedSmoothing(model, sigma=0.05, n=40, batch_size=32)
    preds = smooth.predict(x)
    assert preds.shape == (4,)
    # con poco ruido y un modelo determinista, el voto debe coincidir con el modelo
    with torch.no_grad():
        direct = model(x).argmax(1)
    assert (preds == direct).all()


def test_smoothing_certify_returns_nonnegative_radii(model):
    try:
        import scipy  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("scipy no instalado")
    torch.manual_seed(0)
    x = torch.rand(2, 1, 28, 28)
    smooth = RandomizedSmoothing(model, sigma=0.1, n=100, batch_size=64)
    radii = smooth.certify(x, n=100, alpha=0.001)
    assert radii.shape == (2,)
    assert (radii >= 0).all()
