"""Datasets, cargadores y utilidades de checkpoints.

Punto clave del diseño: los datasets entregan imágenes **sin normalizar**
(en ``[0, 1]``). La normalización ocurre dentro del modelo (ver
``adversarial_playground.models.layers.Normalize``). Así, los ataques
trabajan en el espacio real de píxeles y ``eps`` tiene unidades de intensidad
de píxel (0-1), que es lo que puede percibir/restringir un atacante real.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Media/desviación típica por dataset (coinciden con las de los modelos).
DATASET_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

# Nombres de las clases, para etiquetar las visualizaciones.
CLASS_NAMES: dict[str, list[str]] = {
    "mnist": [str(d) for d in range(10)],
    "cifar10": [
        "avión",
        "coche",
        "pájaro",
        "gato",
        "ciervo",
        "perro",
        "rana",
        "caballo",
        "barco",
        "camión",
    ],
}

# Arquitectura por defecto para cada dataset.
DEFAULT_MODEL: dict[str, str] = {"mnist": "mnist_cnn", "cifar10": "cifar_resnet"}

AVAILABLE_DATASETS = tuple(DATASET_STATS)


def denormalize(x: Tensor, dataset: str) -> Tensor:
    """Pasa imágenes de ``[0, 1]`` a espacio visual "canónico" para pintar.

    Como aquí no normalizamos en el dataset, para MNIST basta con dejar la
    imagen tal cual; en CIFAR-10 elevamos el contraste de forma suave para
    que las visualizaciones sean legibles (el contraste original es bajo).

    Acepta tanto un lote ``(B, C, H, W)`` como una imagen ``(C, H, W)``.

    Args:
        x: imágenes en ``[0, 1]``.
        dataset: clave de :data:`DATASET_STATS`.
    """
    if dataset == "cifar10":
        # Estiramiento suave de contraste al 2%-98% (por imagen), sin saturar.
        hw = (1, 2) if x.ndim == 3 else (2, 3)
        lo = x.amin(dim=hw, keepdim=True)
        hi = x.amax(dim=hw, keepdim=True)
        return ((x - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return x.clamp(0, 1)


def get_datasets(
    name: str, root: str | Path = "data"
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Devuelve ``(train, test)`` para un dataset soportado.

    Las imágenes se devuelven como tensores en ``[0, 1]`` sin normalizar
    (ver docstring del módulo). Se descargan automáticamente la primera vez
    (MNIST ~11 MB, CIFAR-10 ~170 MB).

    Raises:
        ValueError: si ``name`` no está en :data:`AVAILABLE_DATASETS`.
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset '{name}' no soportado. Disponibles: {AVAILABLE_DATASETS}"
        )
    to_tensor = transforms.ToTensor()
    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=to_tensor)
        test = datasets.MNIST(root, train=False, download=True, transform=to_tensor)
    else:  # cifar10
        train = datasets.CIFAR10(root, train=True, download=True, transform=to_tensor)
        test = datasets.CIFAR10(root, train=False, download=True, transform=to_tensor)
    return train, test


def get_loaders(
    name: str,
    batch_size: int = 128,
    root: str | Path = "data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """DataLoaders de train y test (train barajado, test ordenado).

    ``num_workers=0`` por defecto: en Windows evita problemas de spawn y en
    datasets pequeños no aporta apenas nada.
    """
    train_ds, test_ds = get_datasets(name, root)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    arch: str,
    dataset: str,
    epochs: int,
    clean_acc: float,
    robust_acc: float | None = None,
    extra: dict | None = None,
) -> None:
    """Guarda un checkpoint con metadatos para poder reconstruir el modelo.

    Args:
        model: módulo PyTorch (se guardan sus ``state_dict``).
        path: destino ``.pt``.
        arch: clave de ``MODEL_REGISTRY`` (``"mnist_cnn"``...).
        dataset: dataset con el que se entrenó.
        epochs: épocas de entrenamiento.
        clean_acc: precisión limpia en test (0-1).
        robust_acc: precisión bajo PGD si se evaluó (0-1).
        extra: metadatos adicionales (p. ej. parámetros del AT).
    """
    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "dataset": dataset,
        "epochs": epochs,
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Checkpoint guardado en %s", path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict]:
    """Carga un checkpoint y devuelve ``(modelo_en_eval, metadatos)``.

    El modelo se devuelve en modo ``eval()`` (listo para atacar/evaluar);
    si vas a entrenar, recuerda hacer ``model.train()``.

    Raises:
        FileNotFoundError: si el checkpoint no existe.
        KeyError: si la arquitectura guardada no está en ``MODEL_REGISTRY``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint '{path}'. Entrena uno con "
            f"`python scripts/train.py --dataset mnist` o descarga los oficiales."
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    from adversarial_playground.models import MODEL_REGISTRY  # evita import circular

    arch = payload["arch"]
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"Arquitectura '{arch}' desconocida; ¿versión antigua del repo?")
    model = MODEL_REGISTRY[arch]()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
