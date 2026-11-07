"""Curvas: confianza vs eps y precisión robusta vs eps."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import BaseAttack
from adversarial_playground.utils.data import CLASS_NAMES, denormalize


def plot_confidence_vs_eps(
    model: nn.Module,
    attack_factory: Callable[[float], BaseAttack],
    image: Tensor,
    label: int,
    eps_values: list[float],
    dataset: str = "mnist",
    path: str | Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """Confianza del modelo según crece el presupuesto ``eps``.

    Dos curvas: probabilidad (softmax) de la clase verdadera y de la clase
    ganadora adversarial, para cada ``eps``. Es la gráfica que mejor explica
    "el modelo pierde la fe en la respuesta correcta" a medida que el
    atacante gasta presupuesto.

    Args:
        model: modelo objetivo.
        attack_factory: ``eps -> ataque`` (se evalúa una pasada por eps).
        image: imagen ``(1, C, H, W)`` o ``(C, H, W)`` en [0, 1].
        label: clase verdadera.
        eps_values: presupuestos a barrer.
        dataset: nombres de clase.
        path: ruta PNG de salida.

    Returns:
        Ruta del PNG o ``None``.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    class_names = CLASS_NAMES[dataset]
    true_probs, winner_probs = [], []

    for eps in eps_values:
        attack = attack_factory(eps)
        result = attack(image, torch.tensor([label]))
        with torch.no_grad():
            probs = model(result.adv_images).softmax(dim=1)[0]
        true_probs.append(float(probs[label]))
        winner = int(probs.argmax())
        winner_probs.append(float(probs[winner]))

    fig, (ax_img, ax_curve) = plt.subplots(1, 2, figsize=(8.6, 3.4), width_ratios=[1, 2])

    # Imagen junto a la curva, para saber "de qué hablamos"
    disp = denormalize(image[0].detach().cpu(), dataset).permute(1, 2, 0).numpy()
    gray = image.shape[1] == 1
    ax_img.imshow(disp, cmap="gray" if gray else None)
    ax_img.set_title(f"real: {class_names[label]}", fontsize=10)
    ax_img.axis("off")

    ax_curve.plot(eps_values, true_probs, marker="o", label=f"confianza en {class_names[label]}")
    ax_curve.plot(
        eps_values, winner_probs, marker="s", color="#c62828", label="confianza en la predicción"
    )
    ax_curve.axhline(0.5, color="gray", lw=0.6, ls="--")
    ax_curve.set_xlabel("presupuesto del atacante eps (L∞)")
    ax_curve.set_ylabel("probabilidad softmax")
    ax_curve.set_title("¿Cuánto presupuesto necesita un atacante?", fontsize=11)
    ax_curve.legend(fontsize=9)
    ax_curve.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    if path is None:
        plt.close(fig)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_robust_curves(
    curves: dict[str, list[dict[str, Any]]],
    title: str = "Precisión robusta vs presupuesto del atacante",
    path: str | Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """Compara curvas ``robust_acc`` vs ``eps`` de varios modelos.

    Args:
        curves: mapeo ``nombre_modelo -> filas de robust_accuracy_sweep``
            (cada fila con claves ``eps`` y ``robust_acc``).
        title: título del gráfico.
        path: ruta PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    palette = ["#1565c0", "#c62828", "#2e7d32", "#ef6c00", "#6a1b9a"]
    for (name, rows), color in zip(curves.items(), palette, strict=False):
        eps = [r["eps"] for r in rows]
        acc = [r["robust_acc"] for r in rows]
        ax.plot(eps, acc, marker="o", label=name, color=color)

    ax.set_xlabel("eps del ataque (L∞)")
    ax.set_ylabel("precisión robusta")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if path is None:
        plt.close(fig)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
