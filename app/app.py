"""Demo interactiva con Gradio: ataca dígitos MNIST en vivo.

Ejecutar desde la raíz del repo::

    pip install ".[app]"
    python app/app.py

Se abre en http://127.0.0.1:7860 con sliders para el ataque, el presupuesto
eps y los pasos. Incluye la pestaña de "robustez": compara el modelo limpio
con el entrenado adversarialmente bajo el mismo ataque.

Para desplegarla gratis en Hugging Face Spaces: crea un Space (SDK Gradio),
sube `app/app.py`, `checkpoints/`, `adversarial_playground/` y un
`requirements.txt` con `torch`, `torchvision`, `gradio`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from adversarial_playground.attacks.factory import build_attack
from adversarial_playground.utils.data import CLASS_NAMES, get_loaders, load_checkpoint
from adversarial_playground.visualization.grids import plot_attack_grid

CKPT_CLEAN = ROOT / "checkpoints" / "mnist_cnn.pt"
CKPT_AT = ROOT / "checkpoints" / "mnist_cnn_adv.pt"

# Carga perezosa (Gradio la hace una vez al arrancar)
_cache: dict[str, object] = {}


def _get_model(which: str):
    key = "at" if which == "modelo AT (robusto)" else "clean"
    if key not in _cache:
        path = CKPT_AT if key == "at" else CKPT_CLEAN
        if not path.exists():
            raise gr.Error(
                f"Falta {path.name}. Entrena con scripts/train.py "
                f"{'--adversarial ' if key == 'at' else ''}primero."
            )
        _cache[key] = load_checkpoint(path)[0]
    return _cache[key]


def _get_test_images():
    if "batch" not in _cache:
        _, loader = get_loaders("mnist", batch_size=64)
        x, y = next(iter(loader))
        _cache["batch"] = (x, y)
    return _cache["batch"]


def run_attack(attack_name: str, eps: float, steps: int, index: int, which_model: str):
    """Genera la rejilla original/perturbación/adversarial para una imagen."""
    model = _get_model(which_model)
    images, labels = _get_test_images()
    index = int(index) % images.shape[0]

    attack = build_attack(attack_name, model, eps=eps, steps=int(steps))
    result = attack(images[index : index + 1], labels[index : index + 1])

    out = plot_attack_grid(
        model,
        attack,
        images[index : index + 1],
        labels[index : index + 1],
        dataset="mnist",
        path=None,
        n=1,
        amplify=10,
    )
    fooled = bool(result.success[0])
    verdict = (
        f"✅ Ataque exitoso — el modelo ahora predice «{CLASS_NAMES['mnist'][int(result.adv_labels[0])]}» "
        f"(perturbación: L∞={result.mean_linf:.3f}, L2={result.mean_l2:.3f})"
        if fooled
        else f"❌ El ataque no logró engañar al modelo (L∞={result.mean_linf:.3f}, L2={result.mean_l2:.3f})"
    )
    return out, verdict


with gr.Blocks(title="Adversarial ML Playground") as demo:
    gr.Markdown(
        """
        # 🎯 Adversarial ML Playground
        Elige una imagen del test de MNIST y un ataque, y observa cómo una
        perturbación **imperceptible** cambia la predicción del modelo.
        """
    )
    with gr.Row():
        with gr.Column():
            attack_name = gr.Dropdown(
                ["fgsm", "bim", "pgd"], value="pgd", label="Ataque"
            )
            eps = gr.Slider(0.01, 0.45, value=0.2, step=0.01, label="eps (presupuesto L∞)")
            steps = gr.Slider(1, 100, value=40, step=1, label="pasos (solo BIM/PGD)")
            index = gr.Slider(0, 63, value=17, step=1, label="índice de la imagen (test)")
            which_model = gr.Radio(
                ["modelo limpio", "modelo AT (robusto)"],
                value="modelo limpio",
                label="Modelo objetivo",
            )
            btn = gr.Button("Atacar", variant="primary")
        with gr.Column():
            grid = gr.Image(label="original | perturbación ×10 | adversarial", interactive=False)
            verdict = gr.Markdown()

    btn.click(run_attack, [attack_name, eps, steps, index, which_model], [grid, verdict])

    gr.Markdown(
        """
        **¿Qué estoy viendo?** La columna central es la perturbación amplificada
        ×10 — en la imagen real es invisible. Prueba: `pgd` con `eps=0.3` sobre
        el modelo limpio (engaña casi siempre) y luego el mismo ataque sobre el
        **modelo AT** (resiste porque se entrenó contra PGD). Más teoría en el
        [README](https://github.com/raulp/adversarial-ml-playground).
        """
    )

if __name__ == "__main__":
    demo.launch()
