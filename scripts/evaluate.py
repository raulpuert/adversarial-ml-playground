#!/usr/bin/env python
"""Evalúa la robustez de un checkpoint frente a los ataques del zoo.

Ejemplos::

    # Barrido de eps con los tres ataques de gradiente
    python scripts/evaluate.py --checkpoint checkpoints/mnist_cnn.pt \
        --attacks fgsm,bim,pgd --eps 0.05,0.1,0.2,0.3 --csv results/mnist.csv

    # Ataques fuertes sobre un subconjunto (son lentos)
    python scripts/evaluate.py --checkpoint checkpoints/mnist_cnn.pt \
        --attacks deepfool,cw --limit 1000 --csv results/mnist_slow.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse


def main() -> int:
    from adversarial_playground.cli import cmd_evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attacks", default="fgsm,bim,pgd")
    parser.add_argument("--eps", default="0.1,0.2,0.3")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    return cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
