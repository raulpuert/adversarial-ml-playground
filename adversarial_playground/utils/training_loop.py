"""Bucle de entrenamiento compartido por la CLI y ``scripts/train.py``.

Soporta entrenamiento limpio y entrenamiento adversarial (PGD-AT de Madry).
Al terminar evalúa precisión limpia y robusta, guarda el checkpoint con
metadatos y devuelve un resumen imprimible.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from adversarial_playground.attacks.factory import build_attack
from adversarial_playground.defenses.training import AdvTrainingConfig, adversarial_loss
from adversarial_playground.evaluation.metrics import evaluate_attack
from adversarial_playground.models import MODEL_REGISTRY
from adversarial_playground.utils.data import (
    DEFAULT_MODEL,
    get_loaders,
    save_checkpoint,
)
from adversarial_playground.utils.seeding import get_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuración completa de un entrenamiento.

    Attributes:
        dataset: ``"mnist"`` o ``"cifar10"``.
        arch: clave de ``MODEL_REGISTRY`` (por defecto, la del dataset).
        epochs: épocas de entrenamiento (3 MNIST, 10 CIFAR si no se indica).
        batch_size: tamaño de batch.
        lr: tasa de aprendizaje (Adam).
        adversarial: activar entrenamiento adversarial (PGD-AT).
        adv: configuración del PGD generador de adversariales.
        mix_ratio: proporción limpia/adversarial (0 = Madry puro).
        out: ruta del checkpoint de salida.
        seed: semilla global.
        device: dispositivo (auto si ``None``).
        num_workers: workers del DataLoader.
        robust_eval_limit: nº de imágenes para la evaluación robusta final.
    """

    dataset: str = "mnist"
    arch: str | None = None
    epochs: int | None = None
    batch_size: int = 128
    lr: float = 1e-3
    adversarial: bool = False
    adv: AdvTrainingConfig = field(default_factory=AdvTrainingConfig)
    mix_ratio: float = 0.0
    out: Path | None = None
    seed: int = 42
    device: str | None = None
    num_workers: int = 0
    robust_eval_limit: int = 2000


def run_training(cfg: TrainConfig) -> dict:
    """Entrena el modelo y guarda el checkpoint. Devuelve un resumen."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    arch = cfg.arch or DEFAULT_MODEL[cfg.dataset]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Arquitectura '{arch}' desconocida: {list(MODEL_REGISTRY)}")
    epochs = cfg.epochs or (3 if cfg.dataset == "mnist" else 10)

    train_loader, test_loader = get_loaders(
        cfg.dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    model = MODEL_REGISTRY[arch]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_params = sum(p.numel() for p in model.parameters())
    mode = "PGD-AT (Madry)" if cfg.adversarial else "limpio"
    logger.info(
        "Entrenando %s en %s · %d parámetros · %d épocas · modo %s",
        arch,
        device,
        n_params,
        epochs,
        mode,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        running_loss, running_acc_adv, batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"época {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if cfg.adversarial:
                loss, stats = adversarial_loss(
                    model, images, labels, cfg.adv, mix_ratio=cfg.mix_ratio
                )
                running_acc_adv += stats["acc_adv"]
            else:
                loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            batches += 1
            pbar.set_postfix(loss=f"{running_loss / batches:.4f}")

        dt = time.perf_counter() - t0
        msg = f"época {epoch}: loss={running_loss / batches:.4f} ({dt:.0f}s)"
        if cfg.adversarial:
            msg += f" · acc sobre adversariales={running_acc_adv / batches:.3f}"
        logger.info(msg)

    # ---- evaluación final: limpio en todo el test, robusto en un subconjunto
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model(images).argmax(1) == labels).sum())
            total += labels.shape[0]
    clean_acc = correct / total
    logger.info("Precisión limpia (test completo): %.2f%%", 100 * clean_acc)
    robust_eps = cfg.adv.eps if cfg.adversarial else 0.3
    # Medimos robustez SIEMPRE (también los modelos limpios): así ambos
    # checkpoints son comparables en las tablas del README.
    attack = build_attack("pgd", model, eps=robust_eps, steps=min(20, max(7, cfg.adv.steps)))
    evaluation = evaluate_attack(
        model, attack, test_loader, max_samples=cfg.robust_eval_limit, show_progress=True
    )
    robust_acc = evaluation.robust_acc
    logger.info(
        "Precisión robusta PGD(eps=%.2f, %d pasos) sobre %d imágenes: %.2f%%",
        robust_eps,
        attack.steps,  # type: ignore[attr-defined]
        evaluation.n_samples,
        100 * robust_acc,
    )

    out = cfg.out or Path(
        f"checkpoints/{cfg.dataset}_{arch}{'_adv' if cfg.adversarial else ''}.pt"
    )
    save_checkpoint(
        model,
        out,
        arch=arch,
        dataset=cfg.dataset,
        epochs=epochs,
        clean_acc=clean_acc,
        robust_acc=robust_acc,
        extra={
            "adversarial": cfg.adversarial,
            "adv_eps": cfg.adv.eps,
            "adv_alpha": cfg.adv.alpha,
            "adv_steps": cfg.adv.steps,
            "mix_ratio": cfg.mix_ratio,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
    )
    return {"out": str(out), "clean_acc": clean_acc, "robust_acc": robust_acc, "arch": arch}


def load_model_from_checkpoint(
    path: str | Path, device: str | None = None
) -> tuple[nn.Module, dict]:
    """Carga un checkpoint al dispositivo correcto y lo deja en ``eval()``."""
    from adversarial_playground.utils.data import load_checkpoint

    model, payload = load_checkpoint(path, map_location="cpu")
    model.to(get_device(device))
    return model, payload
