#!/usr/bin/env python
"""Genera los assets del README (imágenes, GIF, curvas) y los CSV de resultados.

Requiere los checkpoints entrenados::

    checkpoints/mnist_cnn.pt      (modelo limpio)
    checkpoints/mnist_cnn_adv.pt  (modelo PGD-AT)
    checkpoints/cifar_resnet.pt   (opcional, para la rejilla CIFAR)

Ejecuta::

    python scripts/train.py --dataset mnist
    python scripts/train.py --dataset mnist --adversarial
    python scripts/train.py --dataset cifar10
    python scripts/make_assets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import logging

import torch

from adversarial_playground.attacks.deepfool import DeepFool
from adversarial_playground.attacks.factory import build_attack
from adversarial_playground.attacks.fgsm import FGSM
from adversarial_playground.evaluation.metrics import evaluate_attack, robust_accuracy_sweep
from adversarial_playground.utils.data import get_loaders, load_checkpoint
from adversarial_playground.utils.seeding import get_device, set_seed
from adversarial_playground.visualization.curves import (
    plot_confidence_vs_eps,
    plot_robust_curves,
)
from adversarial_playground.visualization.gif import make_epsilon_gif
from adversarial_playground.visualization.grids import plot_attack_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("make_assets")

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

# Muestras por defecto (equilibrio estadístico/tiempo en CPU)
N_GRIDS = 6
N_SWEEP = 2000
N_SLOW = 500
N_CW = 256  # C&W es caro: menos muestras, más presupuesto por muestra
N_DEFENSE = 1000

EPS_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45]

# Presupuesto "serio" de C&W: la búsqueda binaria necesita rondas suficientes
# para alcanzar valores de c que superen márgenes de logit grandes.
CW_KWARGS = dict(binary_search_steps=5, max_iter=300, learning_rate=0.02)


def pick_samples(model, loader, attack, n, device):
    """Primeras ``n`` imágenes del test que el modelo acierta y el ataque engaña.

    Se usa solo para las **rejillas ilustrativas** (el objetivo del asset es
    mostrar el fenómeno, no su frecuencia — eso lo miden los CSV). Selección
    determinista: orden fijo del test + semilla.

    Detalle de consistencia: ``plot_attack_grid`` re-ejecuta el ataque sobre
    el lote que recibe, así que aquí recolectamos candidatos de más y
    **re-verificamos el éxito sobre ese lote exacto** antes de devolverlo —
    evita filas "verdes" (ataque fallido) por muestras al límite.
    """
    model.eval()
    cand_imgs, cand_labels = [], []
    target = 3 * n  # candidatos extra por si al re-verificar cae alguno
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_ok = model(images).argmax(1) == labels
        if not clean_ok.any():
            continue
        result = attack(images[clean_ok], labels[clean_ok])
        for i in range(images[clean_ok].shape[0]):
            if bool(result.success[i]):
                cand_imgs.append(images[clean_ok][i])
                cand_labels.append(labels[clean_ok][i])
                if len(cand_imgs) >= target:
                    break
        if len(cand_imgs) >= target:
            break

    if not cand_imgs:
        raise RuntimeError("Ninguna muestra exitosa encontrada para la rejilla")

    imgs = torch.stack(cand_imgs)
    labels_t = torch.stack(cand_labels)
    # Verificación doble sobre el lote exacto que verá plot_attack_grid:
    # las muestras "al filo" (p. ej. 49.4% vs 50.6%) pueden oscilar entre
    # pasadas por no-determinismo float del backward multihilo; exigir
    # éxito en DOS pasadas elimina casi todas.
    first = attack(imgs, labels_t).success
    second = attack(imgs, labels_t).success
    ok = first & second
    return imgs[ok][:n], labels_t[ok][:n]


def make_grids(clean_model, at_model, test_loader, device):
    logger.info("→ Rejillas de ataques (FGSM, PGD, DeepFool)")

    fgsm = FGSM(clean_model, eps=0.2)
    imgs, labels = pick_samples(clean_model, test_loader, fgsm, N_GRIDS, device)
    plot_attack_grid(
        clean_model, fgsm, imgs, labels, "mnist", ASSETS / "fgsm_grid.png", n=N_GRIDS
    )

    pgd = build_attack("pgd", clean_model, eps=0.3, steps=40)
    imgs, labels = pick_samples(clean_model, test_loader, pgd, N_GRIDS, device)
    plot_attack_grid(
        clean_model, pgd, imgs, labels, "mnist", ASSETS / "pgd_grid.png", n=N_GRIDS
    )

    # DeepFool sobre el modelo limpio: perturbación minúscula
    deepfool = DeepFool(clean_model)
    imgs, labels = pick_samples(clean_model, test_loader, deepfool, N_GRIDS, device)
    plot_attack_grid(
        clean_model, deepfool, imgs, labels, "mnist", ASSETS / "deepfool_grid.png", n=N_GRIDS
    )

    # El modelo AT resiste el mismo PGD: comparación visual
    if at_model is not None:
        imgs_at, labels_at = pick_samples(at_model, test_loader, pgd, N_GRIDS, device)
        plot_attack_grid(
            at_model,
            pgd,
            imgs_at,
            labels_at,
            "mnist",
            ASSETS / "pgd_grid_at_model.png",
            n=N_GRIDS,
        )


def make_confidence_curve(clean_model, test_loader, device):
    logger.info("→ Curva de confianza vs eps")
    # Buscamos una imagen bien clasificada con alta confianza a la que
    # además FGSM(0.15) SÍ engaña: la curva debe mostrar el volteo, no solo
    # la degradación de confianza.
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            probs = clean_model(images).softmax(1)
            conf, preds = probs.max(1)
            ok = preds == labels
        if not ok.any():
            continue
        result = FGSM(clean_model, eps=0.15)(images[ok], labels[ok])
        flipped = result.success
        if not flipped.any():
            continue
        idx_candidates = torch.where(ok)[0][flipped]
        conf_candidates = conf[idx_candidates]
        idx = int(idx_candidates[int(conf_candidates.argmax())])
        image, label = images[idx], int(labels[idx])
        break
    attack_factory = lambda eps: FGSM(clean_model, eps=eps)  # noqa: E731
    plot_confidence_vs_eps(
        clean_model,
        attack_factory,
        image,
        label,
        [round(0.02 * i, 2) for i in range(16)],
        "mnist",
        ASSETS / "confidence_vs_eps.png",
    )


def make_hero_gif(clean_model, test_loader, device):
    logger.info("→ GIF del README (barrido de eps con FGSM)")

    def good_sample():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                probs = clean_model(images).softmax(1)
                conf, preds = probs.max(1)
            for i in range(images.shape[0]):
                if preds[i] == labels[i] and conf[i] > 0.95:
                    result = FGSM(clean_model, eps=0.12)(images[i : i + 1], labels[i : i + 1])
                    if bool(result.success[0]):
                        return images[i], int(labels[i])
        return images[0], int(labels[0])

    image, label = good_sample()
    make_epsilon_gif(
        clean_model,
        image,
        label,
        lambda eps: FGSM(clean_model, eps=eps),
        dataset="mnist",
        path=ASSETS / "attack_demo.gif",
        fps=5,
    )


def write_attacks_csv(clean_model, test_loader, device):
    logger.info("→ CSV: zoo de ataques sobre el modelo limpio")
    rows = []
    for name, eps_list in (("fgsm", EPS_SWEEP), ("bim", EPS_SWEEP), ("pgd", EPS_SWEEP)):
        for eps in eps_list:
            attack = build_attack(name, clean_model, eps=eps)
            ev = evaluate_attack(
                clean_model, attack, test_loader, max_samples=N_SWEEP, show_progress=False
            )
            rows.append(_row(name, eps, ev))
            logger.info("   %s eps=%.2f → robust %.2f%%", name, eps, 100 * ev.robust_acc)

    for name in ("deepfool", "cw"):
        if name == "cw":
            from adversarial_playground.attacks.cw import CarliniWagnerL2

            attack = CarliniWagnerL2(clean_model, **CW_KWARGS)
            n_samples = N_CW
        else:
            attack = build_attack(name, clean_model)
            n_samples = N_SLOW
        ev = evaluate_attack(
            clean_model, attack, test_loader, max_samples=n_samples, show_progress=False
        )
        rows.append(_row(name, "", ev))
        logger.info("   %s → robust %.2f%%", name, 100 * ev.robust_acc)

    _write_csv(RESULTS / "mnist_attacks.csv", rows)


def write_at_comparison_csv(clean_model, at_model, test_loader, device):
    logger.info("→ CSV: modelo limpio vs PGD-AT")
    curves = {}
    rows = []

    def factory(model):
        return lambda eps: build_attack("pgd", model, eps=eps, steps=40)

    for name, model in (("limpio", clean_model), ("pgd_at", at_model)):
        sweep = robust_accuracy_sweep(
            model, factory(model), EPS_SWEEP, test_loader, max_samples=N_SWEEP
        )
        curves[name] = sweep
        for r in sweep:
            rows.append({"model": name, **r})
            logger.info("   %s eps=%.2f → robust %.2f%%", name, r["eps"], 100 * r["robust_acc"])

    plot_robust_curves(
        curves,
        title="Robustez: modelo limpio vs entrenamiento adversarial (PGD)",
        path=ASSETS / "robust_curves.png",
    )
    _write_csv(RESULTS / "mnist_at_comparison.csv", rows)


def write_defenses_csv(clean_model, test_loader, device):
    logger.info("→ CSV: defensas de transformación de entrada")
    from adversarial_playground.defenses.transforms import (
        BitDepthReduction,
        GaussianNoise,
        JPEGCompression,
        TotalVarianceMinimization,
    )

    pgd = build_attack("pgd", clean_model, eps=0.3, steps=40)
    defenses = {
        "sin_defensa": None,
        "bitdepth_3bits": BitDepthReduction(bits=3),
        "bitdepth_5bits": BitDepthReduction(bits=5),
        "jpeg_q75": JPEGCompression(quality=75),
        "jpeg_q90": JPEGCompression(quality=90),
        "ruido_sigma0.1": GaussianNoise(sigma=0.1),
        "tvm": TotalVarianceMinimization(),
    }
    rows = []
    for name, defense in defenses.items():
        if defense is None:
            ev = evaluate_attack(
                clean_model, pgd, test_loader, max_samples=N_DEFENSE, show_progress=False
            )
        else:
            from adversarial_playground.evaluation.metrics import evaluate_transform_defense

            ev = evaluate_transform_defense(
                clean_model, defense, pgd, test_loader, max_samples=N_DEFENSE
            )
        rows.append(
            {
                "defensa": name,
                "ataque": "pgd_eps0.3",
                "n": ev.n_samples,
                "clean_acc": round(ev.clean_acc, 4),
                "robust_acc": round(ev.robust_acc, 4),
                "attack_success": round(ev.attack_success, 4),
            }
        )
        logger.info(
            "   %-18s clean %.2f%% | robust %.2f%%",
            name,
            100 * ev.clean_acc,
            100 * ev.robust_acc,
        )
    _write_csv(RESULTS / "mnist_defenses.csv", rows)


def maybe_cifar_grid(device):
    ckpt = ROOT / "checkpoints" / "cifar_resnet.pt"
    if not ckpt.exists():
        logger.warning("Sin checkpoint CIFAR; se omite la rejilla CIFAR")
        return
    model, _ = load_checkpoint(ckpt)
    model.to(device)
    _, test_loader = get_loaders("cifar10", batch_size=256)
    pgd = build_attack("pgd", model, eps=8 / 255, steps=20)
    imgs, labels = pick_samples(model, test_loader, pgd, N_GRIDS, device)
    plot_attack_grid(model, pgd, imgs, labels, "cifar10", ASSETS / "cifar_pgd_grid.png", n=N_GRIDS)


def _row(attack, eps, ev):
    return {
        "attack": attack,
        "eps": eps,
        "n": ev.n_samples,
        "clean_acc": round(ev.clean_acc, 4),
        "robust_acc": round(ev.robust_acc, 4),
        "attack_success": round(ev.attack_success, 4),
        "mean_l2": round(ev.mean_l2, 4),
        "mean_linf": round(ev.mean_linf, 4),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("   guardado %s", path)


def main() -> int:
    set_seed(42)
    device = get_device()
    ASSETS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    clean_model, meta_clean = load_checkpoint(ROOT / "checkpoints" / "mnist_cnn.pt")
    clean_model.to(device)
    at_path = ROOT / "checkpoints" / "mnist_cnn_adv.pt"
    at_model = None
    if at_path.exists():
        at_model, _ = load_checkpoint(at_path)
        at_model.to(device)
    else:
        logger.warning("Sin %s: se omiten los assets del modelo AT", at_path)

    _, test_loader = get_loaders("mnist", batch_size=128)

    make_grids(clean_model, at_model, test_loader, device)
    make_confidence_curve(clean_model, test_loader, device)
    make_hero_gif(clean_model, test_loader, device)
    write_attacks_csv(clean_model, test_loader, device)
    if at_model is not None:
        write_at_comparison_csv(clean_model, at_model, test_loader, device)
    write_defenses_csv(clean_model, test_loader, device)
    maybe_cifar_grid(device)

    logger.info("Assets listos en %s y resultados en %s", ASSETS, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
