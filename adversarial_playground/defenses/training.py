"""Entrenamiento adversarial (Madry et al., 2017).

La defensa más sólida conocida: en lugar de entrenar solo con imágenes
limpias, entrena (o preentrena) con **adversariales generados al vuelo**
con PGD. El modelo aprende una función robusta en toda la bola de radio
``eps``, no solo en los puntos del dataset.

Es caro (cada batch exige ``steps`` backward passes extra) y baja algo la
precisión limpia — pero es el único método empírico que aguanta ataques
adaptados sin trucos de ofuscación de gradientes.

Referencias:
    * Madry et al., "Towards Deep Learning Models Resistant to Adversarial
      Attacks": https://arxiv.org/abs/1706.06083
    * TRADES (compromiso exactitud/robustez): https://arxiv.org/abs/1901.08573
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from adversarial_playground.attacks.pgd import PGD


@dataclass
class AdvTrainingConfig:
    """Configuración del ataque PGD usado para generar adversariales de entrenamiento.

    Attributes:
        eps: radio L∞ de la bola de entrenamiento (MNIST: 0.3 como en Madry).
        alpha: paso PGD (Madry usa eps/2.5 ≈ 0.01... realmente 0.01 con eps 0.3;
            ``eps/4`` es una regla razonable y algo más agresiva).
        steps: pasos PGD por batch (Madry usa 40; 7 es el mínimo útil).
        random_start: arranque aleatorio dentro de la bola (recomendado).
    """

    eps: float = 0.3
    alpha: float = 0.075
    steps: int = 7
    random_start: bool = True


def generate_pgd_batch(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    config: AdvTrainingConfig,
) -> Tensor:
    """Genera adversariales PGD para un batch de entrenamiento.

    Separa la generación del ataque del resto de la lógica para que sea
    trivial cambiar PGD por otro ataque (p. ej. FGSM en *free adversarial
    training*) o añadir tu propia variante.
    """
    attack = PGD(
        model,
        eps=config.eps,
        alpha=config.alpha,
        steps=config.steps,
        random_start=config.random_start,
    )
    return attack(images, labels).adv_images


def adversarial_loss(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    config: AdvTrainingConfig,
    mix_ratio: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Pérdida de entrenamiento adversarial de un batch.

    Modos soportados:

    * ``mix_ratio=0.0`` — **Madry puro**: CE solo sobre los adversariales
      (el estándar, más robusto).
    * ``mix_ratio>0`` — **modo mixto**: la primera mitad del batch se usa
      limpia y la segunda mitad adversarial (compromiso entre precisión
      limpia y robustez; ``mix_ratio=0.5`` en esencia).

    Returns:
        ``(loss, stats)`` con ``acc_adv`` (precisión sobre los adversariales)
        para monitorizar la robustez durante el entrenamiento.
    """
    model.train()

    if mix_ratio <= 0:
        adv_images = generate_pgd_batch(model, images, labels, config)
        loss = torch.nn.functional.cross_entropy(model(adv_images), labels)
        stats = {"loss": float(loss.detach()), "acc_adv": _batch_acc(model, adv_images, labels)}
        return loss, stats

    # Modo mixto: primera mitad limpia, segunda mitad adversarial.
    half = max(1, images.shape[0] // 2)
    adv_images = generate_pgd_batch(model, images[half:], labels[half:], config)
    loss_clean = torch.nn.functional.cross_entropy(model(images[:half]), labels[:half])
    loss_adv = torch.nn.functional.cross_entropy(model(adv_images), labels[half:])
    loss = 0.5 * (loss_clean + loss_adv)
    stats = {
        "loss": float(loss.detach()),
        "acc_adv": _batch_acc(model, adv_images, labels[half:]),
    }
    return loss, stats


def _batch_acc(model: nn.Module, images: Tensor, labels: Tensor) -> float:
    if images.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    return float((preds == labels).float().mean().item())
