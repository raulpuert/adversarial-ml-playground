#!/usr/bin/env python
"""Entrena un modelo limpio o con entrenamiento adversarial (PGD-AT).

Ejemplos::

    # Modelo limpio de MNIST (3 épocas, ~5 min en CPU)
    python scripts/train.py --dataset mnist

    # Entrenamiento adversarial de Madry (más lento: PGD de 7 pasos por batch)
    python scripts/train.py --dataset mnist --adversarial

    # CIFAR-10 con ResNet ligera
    python scripts/train.py --dataset cifar10 --epochs 12
"""

import sys
from pathlib import Path

# Permite ejecutar el script desde el raíz del repo sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging


def main() -> int:
    from adversarial_playground.cli import _add_train_args, cmd_train

    parser = argparse.ArgumentParser(description=__doc__)
    _add_train_args(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return cmd_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
