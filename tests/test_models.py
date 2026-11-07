"""Tests de las arquitecturas y la capa de normalización."""

import torch

from adversarial_playground.models import MODEL_REGISTRY, CifarResNet, MnistCNN, Normalize


def test_mnist_cnn_output_shape():
    model = MnistCNN()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert model.num_classes == 10


def test_mnist_cnn_param_count_reasonable():
    n = sum(p.numel() for p in MnistCNN().parameters())
    assert 100_000 < n < 5_000_000


def test_cifar_resnet_output_shape():
    model = CifarResNet()
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_cifar_resnet_widths_configurable():
    model = CifarResNet(widths=(16, 32, 64))
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 10)
    n_small = sum(p.numel() for p in model.parameters())
    n_big = sum(p.numel() for p in CifarResNet().parameters())
    assert n_small < n_big


def test_normalize_applies_mean_std():
    norm = Normalize(mean=(0.5,), std=(0.25,))
    x = torch.full((1, 1, 4, 4), 0.5)
    assert torch.allclose(norm(x), torch.zeros_like(x))
    x2 = torch.full((1, 1, 4, 4), 0.75)
    assert torch.allclose(norm(x2), torch.full_like(x2, 1.0))


def test_registry_has_all_models():
    assert set(MODEL_REGISTRY) == {"mnist_cnn", "cifar_resnet"}
