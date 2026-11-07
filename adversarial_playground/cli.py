"""CLI del playground: ``advml <comando>`` o ``python -m adversarial_playground.cli``.

Comandos::

    advml train     --dataset mnist --epochs 3 [--adversarial]
    advml demo      --checkpoint checkpoints/mnist_cnn.pt --attack pgd --eps 0.3
    advml evaluate  --checkpoint checkpoints/mnist_cnn.pt --attacks fgsm,pgd
    advml gif       --checkpoint checkpoints/mnist_cnn.pt --index 17
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    p.add_argument("--model", default=None, help="arquitectura (por defecto, la del dataset)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adversarial", action="store_true", help="entrenamiento adversarial PGD-AT")
    p.add_argument("--adv-eps", type=float, default=0.3)
    p.add_argument("--adv-steps", type=int, default=7)
    p.add_argument("--mix-ratio", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)


def cmd_train(args: argparse.Namespace) -> int:
    from adversarial_playground.defenses.training import AdvTrainingConfig
    from adversarial_playground.utils.training_loop import TrainConfig, run_training

    cfg = TrainConfig(
        dataset=args.dataset,
        arch=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adversarial=args.adversarial,
        adv=AdvTrainingConfig(eps=args.adv_eps, alpha=args.adv_eps / 4, steps=args.adv_steps),
        mix_ratio=args.mix_ratio,
        out=Path(args.out) if args.out else None,
        seed=args.seed,
        device=args.device,
    )
    summary = run_training(cfg)
    print(f"Checkpoint: {summary['out']}")
    print(f"Precisión limpia: {summary['clean_acc']:.2%}")
    if summary["robust_acc"] is not None:
        print(f"Precisión robusta (PGD): {summary['robust_acc']:.2%}")
    return 0


def _load_model_and_test(args: argparse.Namespace):
    """Modelo + metadata + loader de test según los argumentos comunes."""
    from adversarial_playground.utils.data import get_loaders, load_checkpoint
    from adversarial_playground.utils.seeding import get_device, set_seed

    set_seed(args.seed)
    model, payload = load_checkpoint(args.checkpoint)
    device = get_device(args.device)
    model.to(device)
    _, test_loader = get_loaders(payload["dataset"], batch_size=args.batch_size)
    return model, payload, test_loader, device


def cmd_demo(args: argparse.Namespace) -> int:
    """Rejilla de un ataque sobre las primeras imágenes del test."""
    from adversarial_playground.attacks.factory import build_attack
    from adversarial_playground.visualization.grids import plot_attack_grid

    model, payload, test_loader, _ = _load_model_and_test(args)
    images, labels = next(iter(test_loader))
    images, labels = images.to(next(model.parameters()).device), labels.to(
        next(model.parameters()).device
    )
    attack = build_attack(args.attack, model, eps=args.eps, steps=args.steps)
    out = plot_attack_grid(
        model,
        attack,
        images,
        labels,
        dataset=payload["dataset"],
        path=args.out,
        n=args.n,
    )
    assert out is not None
    print(f"Figura guardada en {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Tabla de métricas de robustez por ataque (y CSV opcional)."""
    import csv

    from adversarial_playground.attacks.factory import build_attack
    from adversarial_playground.evaluation.metrics import evaluate_attack

    model, payload, test_loader, _ = _load_model_and_test(args)
    eps_values = [float(e) for e in args.eps.split(",")]

    rows = []
    for name in [a.strip() for a in args.attacks.split(",")]:
        for eps in eps_values:
            attack = build_attack(name, model, eps=eps, steps=args.steps)
            ev = evaluate_attack(
                model, attack, test_loader, max_samples=args.limit, show_progress=not args.quiet
            )
            rows.append(
                {
                    "checkpoint": args.checkpoint,
                    "dataset": payload["dataset"],
                    "attack": name,
                    "eps": eps,
                    "n_samples": ev.n_samples,
                    "clean_acc": round(ev.clean_acc, 4),
                    "robust_acc": round(ev.robust_acc, 4),
                    "attack_success": round(ev.attack_success, 4),
                    "mean_l2": round(ev.mean_l2, 4),
                    "mean_linf": round(ev.mean_linf, 4),
                }
            )
            print(
                f"{name:>9} eps={eps:<5} → robust_acc={ev.robust_acc:.2%}  "
                f"(éxito {ev.attack_success:.2%}, ‖δ‖₂̄={ev.mean_l2:.3f})"
            )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV guardado en {path}")
    return 0


def cmd_gif(args: argparse.Namespace) -> int:
    """GIF del barrido de eps sobre una imagen del test."""
    from adversarial_playground.attacks.factory import build_attack
    from adversarial_playground.visualization.gif import make_epsilon_gif

    model, payload, test_loader, _ = _load_model_and_test(args)
    images, labels = next(iter(test_loader))
    if args.index >= images.shape[0]:
        print(f"--index {args.index} fuera de rango (batch de {images.shape[0]})")
        return 1
    image = images[args.index].to(next(model.parameters()).device)
    label = int(labels[args.index])
    out = make_epsilon_gif(
        model,
        image,
        label,
        lambda eps: build_attack(args.attack, model, eps=eps),
        dataset=payload["dataset"],
        path=args.out,
        fps=args.fps,
    )
    print(f"GIF guardado en {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advml",
        description="Playground de Machine Learning Adversarial (ataques y defensas).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="logs de depuración")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="entrena un modelo (limpio o PGD-AT)")
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_demo = sub.add_parser("demo", help="rejilla original/perturbación/adversarial")
    p_demo.add_argument("--checkpoint", required=True)
    p_demo.add_argument("--attack", default="pgd", choices=("fgsm", "bim", "pgd"))
    p_demo.add_argument("--eps", type=float, default=0.2)
    p_demo.add_argument("--steps", type=int, default=40)
    p_demo.add_argument("--n", type=int, default=6)
    p_demo.add_argument("--batch-size", type=int, default=64)
    p_demo.add_argument("--out", default="outputs/demo.png")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--device", default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="métricas de robustez por ataque")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--attacks", default="fgsm,bim,pgd")
    p_eval.add_argument("--eps", default="0.1,0.2,0.3")
    p_eval.add_argument("--steps", type=int, default=40)
    p_eval.add_argument("--limit", type=int, default=None, help="máx. imágenes (útil p/ slow)")
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--csv", default=None)
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--device", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_gif = sub.add_parser("gif", help="GIF de confianza vs eps sobre una imagen")
    p_gif.add_argument("--checkpoint", required=True)
    p_gif.add_argument("--attack", default="fgsm", choices=("fgsm", "bim", "pgd"))
    p_gif.add_argument("--index", type=int, default=17, help="índice de la imagen en el test")
    p_gif.add_argument("--eps", type=float, default=0.3)
    p_gif.add_argument("--fps", type=int, default=4)
    p_gif.add_argument("--batch-size", type=int, default=64)
    p_gif.add_argument("--out", default="outputs/attack_demo.gif")
    p_gif.add_argument("--seed", type=int, default=42)
    p_gif.add_argument("--device", default=None)
    p_gif.set_defaults(func=cmd_gif)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
