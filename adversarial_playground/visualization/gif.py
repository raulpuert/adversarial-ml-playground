"""GIF animado: el ataque barrido sobre ``eps``, cuadro a cuadro.

Genera el *hero asset* del README: para cada valor de ``eps``, muestra la
imagen adversarial junto al gráfico de confianza por clase. Al reproducir
la secuencia se ve cómo la predicción "salta" de la clase verdadera a otra
a medida que el atacante gasta presupuesto.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from adversarial_playground.attacks.base import BaseAttack
from adversarial_playground.utils.data import CLASS_NAMES, denormalize


def _frame_image(
    model: nn.Module,
    adv_image: Tensor,
    clean_image: Tensor,
    label: int,
    dataset: str,
    eps: float,
    attack_name: str,
) -> Image.Image:
    """Un cuadro del GIF: imagen adversarial + barras de confianza."""
    class_names = CLASS_NAMES[dataset]
    with torch.no_grad():
        probs = model(adv_image.unsqueeze(0)).softmax(dim=1)[0].detach().cpu()
        clean_probs = model(clean_image.unsqueeze(0)).softmax(dim=1)[0].detach().cpu()

    top5 = torch.topk(probs, 5)
    pred = int(probs.argmax())
    fooled = pred != label

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(6.0, 3.0), width_ratios=[1, 1.4])

    disp = denormalize(adv_image.detach().cpu(), dataset).permute(1, 2, 0).numpy()
    gray = clean_image.shape[0] == 1
    ax_img.imshow(disp, cmap="gray" if gray else None, vmin=0, vmax=1)
    ax_img.axis("off")
    ax_img.set_title(
        f"eps = {eps:.3f}", fontsize=13, color="#c62828" if fooled else "#2e7d32"
    )

    names = [class_names[i] for i in top5.indices]
    colors = ["#c62828" if i == pred and fooled else "#1565c0" for i in top5.indices]
    ax_bar.barh(names[::-1], top5.values.numpy()[::-1], color=colors[::-1])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(labelsize=9)
    ax_bar.set_title(
        f"pred: {class_names[pred]} ({float(probs[pred]):.0%})"
        + ("  ¡ENGAÑADO!" if fooled else ""),
        fontsize=11,
        color="#c62828" if fooled else "#1b5e20",
    )

    clean_pred = int(clean_probs.argmax())
    fig.suptitle(
        f"Ataque {attack_name} · clase real: {class_names[label]} (limpio: "
        f"{class_names[clean_pred]} {float(clean_probs[clean_pred]):.0%})",
        fontsize=10,
    )
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4)
    plt.close(fig)
    return Image.fromarray(buf[..., :3].copy())


def make_epsilon_gif(
    model: nn.Module,
    image: Tensor,
    label: int,
    attack_factory: Callable[[float], BaseAttack],
    eps_values: list[float] | None = None,
    dataset: str = "mnist",
    path: str | Path = "assets/attack_demo.gif",
    fps: int = 4,
    hold_last_frames: int = 4,
) -> Path:
    """Genera un GIF del ataque barrido sobre ``eps``.

    Args:
        model: modelo objetivo.
        image: imagen ``(C, H, W)`` en [0, 1].
        label: clase verdadera.
        attack_factory: ``eps -> ataque`` (p. ej. FGSM o PGD de 1 paso).
        eps_values: presupuestos; por defecto ``[0, 0.01, ..., 0.30]``.
        dataset: nombres de clase.
        path: salida ``.gif``.
        fps: cuadros por segundo.
        hold_last_frames: repite el último cuadro para "pausar" el final.

    Returns:
        Ruta del GIF guardado.
    """
    if image.ndim != 3:
        raise ValueError(f"Espera una imagen (C, H, W); llega {tuple(image.shape)}")
    if eps_values is None:
        eps_values = [round(0.01 * i, 2) for i in range(31)]  # 0.00 … 0.30

    attack_name = type(attack_factory(eps_values[1] if len(eps_values) > 1 else 0.1)).__name__
    clean_image = image.detach().clone()

    frames: list[Image.Image] = []
    for eps in eps_values:
        if eps == 0:
            adv = clean_image
        else:
            result = attack_factory(eps)(clean_image.unsqueeze(0), torch.tensor([label]))
            adv = result.adv_images[0]
        frames.append(
            _frame_image(model, adv, clean_image, label, dataset, eps, attack_name)
        )
    frames += [frames[-1]] * hold_last_frames

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return path
