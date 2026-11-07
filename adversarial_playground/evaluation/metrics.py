"""Métricas de robustez: cómo medir si un modelo aguanta ataques.

Tres preguntas cubren casi todo el análisis adversarial:

1. **¿Cuánto acierta el modelo bajo ataque?** → :func:`evaluate_attack`
   (precisión robusta, tasa de éxito del ataque, tamaño de perturbación).
2. **¿Cómo empeora al crecer ``eps``?** → :func:`robust_accuracy_sweep`
   (curvas de precisión robusta vs. presupuesto).
3. **¿Ayuda una defensa de entrada?** → :func:`evaluate_transform_defense`
   (aplica la transformación antes de predecir y repite la medida).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adversarial_playground.attacks.base import AttackResult, BaseAttack


@dataclass
class AttackEvaluation:
    """Resultado agregado de un ataque sobre un dataloader.

    Attributes:
        n_samples: imágenes evaluadas.
        clean_acc: precisión sobre las imágenes limpias.
        robust_acc: precisión sobre las imágenes adversariales.
        attack_success: tasa de éxito del ataque (solo sobre muestras que el
            modelo clasificaba bien limpias — convención estándar: no se
            puede "engañar" a un modelo que ya fallaba).
        mean_l2 / mean_linf: tamaño medio de la perturbación.
        info: métricas extra agregadas del AttackResult (p. ej. iteraciones).
    """

    n_samples: int = 0
    clean_acc: float = 0.0
    robust_acc: float = 0.0
    attack_success: float = 0.0
    mean_l2: float = 0.0
    mean_linf: float = 0.0
    info: dict[str, float] = field(default_factory=dict)


def _clean_accuracy(model: nn.Module, images: Tensor, labels: Tensor) -> Tensor:
    """Máscara booleana: muestras que el modelo clasifica bien (limpias)."""
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    return preds == labels


def evaluate_attack(
    model: nn.Module,
    attack: BaseAttack,
    loader: DataLoader | Iterator[tuple[Tensor, Tensor]],
    max_samples: int | None = None,
    show_progress: bool = True,
) -> AttackEvaluation:
    """Evalúa un ataque sobre un dataloader completo (o sus primeros batches).

    Args:
        model: modelo objetivo.
        attack: ataque ya configurado (``FGSM(model, eps=0.1)``...).
        loader: dataloader de test (o cualquier iterable de ``(x, y)``).
        max_samples: corta la evaluación tras N imágenes (útil para ataques
            lentos como DeepFool o C&W).
        show_progress: barra de progreso con tqdm.

    Returns:
        :class:`AttackEvaluation` con las métricas agregadas del lote.
    """
    model.eval()
    device = next(model.parameters()).device
    totals = AttackEvaluation()
    correct_clean = 0
    correct_adv = 0
    fooled = 0
    l2_sum = 0.0
    linf_sum = 0.0

    pbar = tqdm(
        loader,
        desc=f"Eval {type(attack).__name__}",
        leave=False,
        disable=not show_progress,
    )
    seen = 0
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        if max_samples is not None and seen + images.shape[0] > max_samples:
            images = images[: max_samples - seen]
            labels = labels[: max_samples - seen]
            if images.shape[0] == 0:
                break

        clean_ok = _clean_accuracy(model, images, labels)
        result: AttackResult = attack(images, labels)

        n = images.shape[0]
        seen += n
        correct_clean += int(clean_ok.sum())
        correct_adv += int((result.adv_labels == labels).sum())
        # Éxito del ataque solo cuenta sobre muestras bien clasificadas limpias
        fooled += int((result.success & clean_ok).sum())
        l2_sum += float(result.l2_norms.sum())
        linf_sum += float(result.linf_norms.sum())

        pbar.set_postfix(
            {
                "rob_acc": f"{correct_adv / seen:.3f}",
                "success": f"{fooled / max(1, correct_clean):.3f}",
            }
        )
        if max_samples is not None and seen >= max_samples:
            break

    totals.n_samples = seen
    totals.clean_acc = correct_clean / max(1, seen)
    totals.robust_acc = correct_adv / max(1, seen)
    totals.attack_success = fooled / max(1, correct_clean)
    totals.mean_l2 = l2_sum / max(1, seen)
    totals.mean_linf = linf_sum / max(1, seen)
    if result.info:
        for key, value in result.info.items():
            totals.info[f"mean_{key}"] = float(value.mean())
    return totals


def robust_accuracy_sweep(
    model: nn.Module,
    attack_factory: Callable[[float], BaseAttack],
    eps_values: list[float],
    loader: DataLoader,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Barrido de ``eps``: precisión robusta para cada presupuesto de ataque.

    Args:
        model: modelo objetivo.
        attack_factory: función ``eps -> BaseAttack`` (p. ej.
            ``lambda eps: PGD(model, eps=eps, alpha=eps/4, steps=40)``).
        eps_values: valores del presupuesto a barrer.
        loader: dataloader de test.
        max_samples: límite de imágenes por punto de la curva.

    Returns:
        Lista de filas ``{"eps", "clean_acc", "robust_acc", "attack_success"}``
        lista para convertir en ``pandas.DataFrame`` o escribir a CSV.
    """
    rows: list[dict[str, Any]] = []
    for eps in eps_values:
        attack = attack_factory(eps)
        evaluation = evaluate_attack(
            model, attack, loader, max_samples=max_samples, show_progress=False
        )
        rows.append(
            {
                "eps": eps,
                "clean_acc": evaluation.clean_acc,
                "robust_acc": evaluation.robust_acc,
                "attack_success": evaluation.attack_success,
            }
        )
    return rows


def evaluate_transform_defense(
    model: nn.Module,
    defense: Callable[[Tensor], Tensor],
    attack: BaseAttack,
    loader: DataLoader,
    max_samples: int | None = None,
) -> AttackEvaluation:
    """Mide cuánto protege una transformación de entrada frente a un ataque.

    Importante para el análisis honesto: la transformación se aplica
    **también a las imágenes limpias** (así medimos el coste de la defensa
    en precisión limpia) y el modelo solo ve imágenes transformadas.
    """
    model.eval()
    device = next(model.parameters()).device
    correct_clean = 0
    correct_adv = 0
    fooled = 0
    seen = 0

    for images, labels in tqdm(loader, desc="Eval defensa", leave=False):
        images, labels = images.to(device), labels.to(device)
        if max_samples is not None and seen + images.shape[0] > max_samples:
            images = images[: max_samples - seen]
            labels = labels[: max_samples - seen]
            if images.shape[0] == 0:
                break

        images_t = defense(images)
        clean_ok = _clean_accuracy(model, images_t, labels)
        result = attack(images, labels)  # el ataque ve la imagen cruda...
        adv_t = defense(result.adv_images)  # ...pero el modelo ve la transformada
        # La predicción final usa la imagen adversarial transformada:
        with torch.no_grad():
            adv_pred = model(adv_t).argmax(dim=1)
        adv_ok = adv_pred == labels

        n = images.shape[0]
        seen += n
        correct_clean += int(clean_ok.sum())
        correct_adv += int(adv_ok.sum())
        fooled += int(((adv_pred != labels) & clean_ok).sum())
        if max_samples is not None and seen >= max_samples:
            break

    return AttackEvaluation(
        n_samples=seen,
        clean_acc=correct_clean / max(1, seen),
        robust_acc=correct_adv / max(1, seen),
        attack_success=fooled / max(1, correct_clean),
    )
