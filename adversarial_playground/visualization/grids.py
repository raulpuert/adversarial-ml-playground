"""Rejillas de visualización: original | perturbación | adversarial."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # este paquete solo guarda figuras a disco
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import AttackResult, BaseAttack
from adversarial_playground.utils.data import CLASS_NAMES, denormalize


def _to_display(img: Tensor, dataset: str) -> Tensor:
    """Imagen (C, H, W) en [0,1] → (H, W, C) lista para ``imshow``."""
    img = denormalize(img.detach().cpu(), dataset)
    return img.permute(1, 2, 0).numpy()


def plot_attack_grid(
    model: nn.Module,
    attack: BaseAttack,
    images: Tensor,
    labels: Tensor,
    dataset: str = "mnist",
    path: str | Path | None = None,
    n: int = 6,
    amplify: float = 12.0,
    dpi: int = 150,
) -> Path | None:
    """Rejilla ``n×3``: original, perturbación amplificada y adversarial.

    Args:
        model: modelo objetivo.
        attack: ataque configurado.
        images: lote de imágenes limpias (se usan las primeras ``n``).
        labels: etiquetas verdaderas.
        dataset: ``"mnist"`` o ``"cifar10"`` (afecta a nombres de clase y
            contraste de visualización).
        path: si se da, guarda la figura ahí.
        n: número de filas (muestras).
        amplify: factor de amplificación del delta para que sea visible.
        dpi: resolución del PNG.

    Returns:
        La ruta del PNG guardado, o ``None`` si no se pasó ``path``.
    """
    result: AttackResult = attack(images, labels)
    model.eval()
    with torch.no_grad():
        clean_probs = model(images).softmax(dim=1)

    class_names = CLASS_NAMES[dataset]
    n = min(n, images.shape[0])
    grayscale = images.shape[1] == 1

    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        # --- columna 1: original ------------------------------------- #
        true_name = class_names[int(labels[i])]
        pred_c = int(clean_probs[i].argmax())
        conf_c = float(clean_probs[i, pred_c].detach())
        cmap0 = "gray" if grayscale else None
        axes[i, 0].imshow(_to_display(images[i], dataset), cmap=cmap0)
        axes[i, 0].set_title(
            f"original\nreal: {true_name} · pred: {class_names[pred_c]} ({conf_c:.0%})",
            fontsize=9,
        )

        # --- columna 2: perturbación amplificada --------------------- #
        delta = (result.adv_images[i] - images[i]) * amplify
        delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
        axes[i, 1].imshow(
            delta.permute(1, 2, 0).detach().cpu().numpy(),
            cmap=None if not grayscale else "viridis",
        )
        axes[i, 1].set_title(
            f"perturbación ×{amplify:.0f}\n"
            f"‖δ‖∞={result.linf_norms[i]:.3f}  ‖δ‖₂={result.l2_norms[i]:.3f}",
            fontsize=9,
        )

        # --- columna 3: adversarial ---------------------------------- #
        pred_a = int(result.adv_labels[i])
        fooled = bool(result.success[i])
        with torch.no_grad():
            adv_probs = model(result.adv_images[i : i + 1]).softmax(dim=1)[0]
        color = "#c62828" if fooled else "#2e7d32"
        axes[i, 2].imshow(
            _to_display(result.adv_images[i], dataset), cmap="gray" if grayscale else None
        )
        axes[i, 2].set_title(
            f"adversarial → {class_names[pred_a]} ({float(adv_probs[pred_a].detach()):.0%})",
            color=color,
            fontsize=9,
        )

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    attack_name = type(attack).__name__
    eps = getattr(attack, "eps", None)
    eps_txt = f" · eps={eps:.4g}" if isinstance(eps, float) else ""
    fig.suptitle(
        f"Ataque {attack_name}{eps_txt}",
        fontsize=12,
        y=1.005 if n > 2 else 1.15,
    )
    fig.tight_layout()

    if path is None:
        plt.close(fig)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
